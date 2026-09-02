from __future__ import annotations

from .checks import bind_checks

from config import breakout_quality as breakout_quality_config

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    COMPARISON_MODE_HARD_FILTER,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
    Path,
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    CONTINUOUS_RANKER_TRAINER_EVENT,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_UPPER_TAIL_RELEVANCE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_R_HUBER_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_R_MSE_PROFILE,
    DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_continuous_ranker_research_spec,
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    DAILY_OPPORTUNITY_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SimpleNamespace,
    StrategyAlignedContinuousTargetSpec,
    TARGET_MANIFEST_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    ast,
    build_daily_full_horizon_opportunity_contract,
    build_daily_full_horizon_pure_mfe_contract,
    build_daily_percentile_targets,
    build_file_manifest,
    build_strategy_aligned_group_targets,
    continuous_ranker_scope_group_ids,
    compute_daily_opportunity_target_batch,
    continuous_ranker_trade_alignment_metrics,
    get_breakout_quality_experiment_profile,
    json,
    load_validated_continuous_target_arrays,
    math,
    np,
    patch,
    pd,
    resolve_continuous_target_dir,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    daily_full_horizon_opportunity_target_from_cached_path,
    daily_full_horizon_pure_mfe_target_from_cached_path,
    daily_opportunity_no_time_target_from_cached_path,
    strategy_aligned_target_from_cached_path,
    tempfile,
)

from .source_index import read_source_ast, read_source_text


def _strategy_c75_uses_experiment_profile(profile_name: str) -> bool:
    """Return whether the current C75 conversion is sourced from this model profile.

    Historical model-only validators must remain valid after a later scientific
    identity legitimately receives a strategy arm.  The invariant is that the
    historical profile itself is not silently reused as C75's source, not that
    the symbol C75 can never exist in the repository.
    """

    from config.strategy_compare import STRATEGY_COMPARE_ARMS, STRATEGY_DL_SOURCES

    arm = dict(STRATEGY_COMPARE_ARMS.get("C75") or {})
    source = dict(STRATEGY_DL_SOURCES.get(str(arm.get("dl_id") or "")) or {})
    return str(source.get("experiment_profile") or "") == str(profile_name)

def validate_breakout_quality_continuous_target_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_TARGET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    contract = spec.contract_payload()
    check("continuous_target_id_is_versioned", STRATEGY_ALIGNED_TARGET_ID, contract["target_id"])
    check(
        "continuous_target_uses_no_split_or_oos_parameters",
        (False, False, "none", "none"),
        (
                    contract["split_derived_parameters"],
                    contract["oos_derived_parameters"],
                    contract["normalization"],
                    contract["clipping"],
                ),
    )
    check(
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
    check(
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
    check(
        "continuous_target_time_penalty_offsets_minimum_move_at_horizon_end",
        (True, horizon, 0.0),
        (bool(late.valid), int(late.opportunity_bar), round(float(late.target_raw_r), 12)),
        tol=1e-9,
    )
    check_true(
        "continuous_target_prefers_early_larger_opportunity",
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
    check(
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
    check(
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
    check(
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

    project_root = Path(__file__).resolve().parents[2]
    builder_source = read_source_text(
        project_root / "services" / "breakout_quality" / "continuous_target_builder.py"
    )
    prepare_source = read_source_text(
        project_root / "services" / "breakout_quality" / "continuous_target_preparation.py"
    )
    catalog_source = read_source_text(project_root / "services" / "audit" / "catalog.py")
    retired_paths = [
        project_root / "tools" / "audit" / "breakout_quality" / name
        for name in (
            "continuous_target.py",
            "no_time_continuous_target.py",
            "qualified_candidate_set.py",
            "target_component_attribution.py",
            "target_time_penalty_ablation.py",
            "target_statistics.py",
            "artifact_primitives.py",
            "point_in_time_scores.py",
        )
    ]
    check_true(
        "continuous_target_builder_is_canonical_service_without_historical_audit_or_strategy_compare_dependency",
        bool(
                    "build_strategy_aligned_target_artifacts" in builder_source
                    and "build_strategy_aligned_no_time_target_artifacts" in builder_source
                    and "historical_research_gate_required" in builder_source
                    and "from tools.audit" not in builder_source
                    and "import tools.audit" not in builder_source
                    and "from filters.breakout_quality.strategy_compare" not in builder_source
                    and "import filters.breakout_quality.strategy_compare" not in builder_source
                    and "target_time_penalty_ablation" not in builder_source
                ),
    )
    check_true(
        "prepare_continuous_target_routes_only_to_canonical_service_builder",
        bool(
                    "services.breakout_quality.continuous_target_builder" in prepare_source
                    and "tools.audit.breakout_quality.continuous_target" not in prepare_source
                    and "approved-workflow-rebuild" not in prepare_source
                ),
    )
    check_true(
        "retired_11c_to_11f_target_audit_implementations_are_not_runtime_or_catalog_commands",
        bool(
                    all(not path.exists() for path in retired_paths)
                    and "audit-continuous-target" not in catalog_source
                    and "audit-qualified-candidate-set" not in catalog_source
                    and "audit-target-attribution" not in catalog_source
                    and "audit-target-time-ablation" not in catalog_source
                    and "audit-no-time-target" not in catalog_source
                    and 'module="services.breakout_quality.point_in_time_audit"' in catalog_source
                ),
    )
    check(
        "continuous_target_output_path_remains_target_version_scoped",
        ("breakout_quality_v1", "continuous_targets", STRATEGY_ALIGNED_TARGET_ID),
        tuple(
                    resolve_continuous_target_dir(
                        Path("/tmp/project"),
                        "breakout_quality_v1",
                    ).parts[-3:]
                ),
    )

    summary["target_id"] = STRATEGY_ALIGNED_TARGET_ID
    summary["builder"] = "services.breakout_quality.continuous_target_builder"
    return results, summary

def validate_breakout_quality_continuous_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    )
    check(
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
    check(
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

    # Capability-oriented contract: every registered continuous-ranker profile is
    # automatically checked here.  A new experiment that only recombines existing
    # target/context/objective capabilities must not require a new synthetic case.
    import config.breakout_quality as breakout_quality_config
    import config.breakout_quality_runtime as continuous_ranker_runtime
    import config.breakout_quality_runtime_resolver as continuous_ranker_runtime_resolver
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
        CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
        CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_UPSIDE,
        CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
        CONTINUOUS_RANKER_PAIRWISE_KIND_NONE,
        CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR,
        CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
        ContinuousRankerObjectivePolicy,
        ContinuousRankerPairWeightPolicy,
        get_continuous_ranker_pair_weight_policy,
        get_profile_enabled_continuous_ranker_training_objectives,
        normalize_continuous_ranker_pair_weight_configuration,
    )
    from config.breakout_quality_runtime_resolver import (
        get_continuous_ranker_execution_recipe,
    )
    from filters.breakout_quality.artifact_dependency_registry import (
        ARTIFACT_CONTINUOUS_TARGET,
        ARTIFACT_DATASET_CORE,
        required_upstream_artifact_types,
    )
    from filters.breakout_quality.predicted_context_artifact import (
        maybe_predicted_context_artifact_spec,
    )
    from filters.breakout_quality.ranker_training_contract import training_semantics

    check(
        "continuous_ranker_runtime_contract_has_dedicated_owner",
        "config.breakout_quality_runtime",
        continuous_ranker_runtime.ContinuousRankerExecutionRecipe.__module__,
    )
    check(
        "continuous_ranker_profile_enabled_objective_membership_derives_from_composition_registry",
        tuple(breakout_quality_config.CONTINUOUS_RANKER_TRAINING_OBJECTIVES),
        tuple(get_profile_enabled_continuous_ranker_training_objectives()),
        note=(
            "config must not maintain a second hand-written objective membership list; "
            "profile eligibility is declared by composition profile_loss_metrics"
        ),
    )
    check_true(
        "continuous_ranker_recipe_resolver_reexports_canonical_config_resolver",
        breakout_quality_config.get_continuous_ranker_execution_recipe
        is continuous_ranker_runtime_resolver.get_continuous_ranker_execution_recipe,
        note=(
            "the acyclic runtime resolver gateway must forward the single canonical "
            "profile-to-recipe resolver owned by config.breakout_quality"
        ),
    )
    runtime_public_names = set(continuous_ranker_runtime.__all__)
    runtime_facade_bypass_imports = []
    project_root = Path(__file__).resolve().parents[2]
    for source_root_name in ("filters", "services"):
        for source_path in sorted((project_root / source_root_name).rglob("*.py")):
            source_tree = ast.parse(source_path.read_text(encoding="utf-8"))
            for node in ast.walk(source_tree):
                if not isinstance(node, ast.ImportFrom) or node.module != "config.breakout_quality":
                    continue
                imported_runtime_names = sorted(
                    alias.name
                    for alias in node.names
                    if (
                        alias.name in runtime_public_names
                        or alias.name == "get_continuous_ranker_execution_recipe"
                    )
                )
                if imported_runtime_names:
                    runtime_facade_bypass_imports.append(
                        f"{source_path.relative_to(project_root).as_posix()}:"
                        + ",".join(imported_runtime_names)
                    )
    check(
        "continuous_ranker_generic_consumers_import_runtime_owner_directly",
        [],
        runtime_facade_bypass_imports,
        note=(
            "filters/services generic consumers must import capability names from "
            "config.breakout_quality_runtime and the profile resolver from "
            "config.breakout_quality_runtime_resolver; config.breakout_quality is not a "
            "generic-consumer runtime import surface"
        ),
    )

    # Pair-weight modification-radius acceptance: policy semantics, scientific contract
    # metadata, trainer execution and report copy all resolve through the runtime registry.
    # A synthetic third policy is injected here without changing trainer/data/report code.
    import torch
    from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
    from services.breakout_quality.train_daily_ranker import _pair_weight_report_extension

    target_diff = torch.tensor([0.4, -0.4], dtype=torch.float32)
    left_context = torch.tensor([0.92, 0.92], dtype=torch.float32)
    right_context = torch.tensor([0.15, 0.15], dtype=torch.float32)
    min_policy = get_continuous_ranker_pair_weight_policy(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY
    )
    winner_policy = get_continuous_ranker_pair_weight_policy(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY
    )
    product_policy = get_continuous_ranker_pair_weight_policy(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY
    )
    conflict_policy = get_continuous_ranker_pair_weight_policy(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY
    )
    check_true(
        "continuous_ranker_pair_weight_registry_preserves_min_directional_product_and_conflict_primitives",
        torch.allclose(
            min_policy.apply(torch, target_diff, left_context, right_context),
            torch.tensor([0.15, 0.15]),
        )
        and torch.allclose(
            winner_policy.apply(torch, target_diff, left_context, right_context),
            torch.tensor([0.92, 0.15]),
        )
        and torch.allclose(
            product_policy.apply(torch, target_diff, left_context, right_context),
            torch.tensor([0.138, 0.138]),
        )
        and torch.allclose(
            conflict_policy.apply(torch, target_diff, left_context, right_context),
            torch.tensor([1.0, 0.15]),
        ),
    )
    conflict_reverse = conflict_policy.apply(
        torch,
        torch.tensor([0.4, -0.4], dtype=torch.float32),
        torch.tensor([0.15, 0.15], dtype=torch.float32),
        torch.tensor([0.92, 0.92], dtype=torch.float32),
    )
    check_true(
        "continuous_ranker_conflict_only_policy_keeps_aligned_pairs_full_and_discounts_only_unsafe_mfe_winner",
        torch.allclose(conflict_reverse, torch.tensor([0.15, 1.0])),
    )
    product_low_low = product_policy.apply(
        torch,
        torch.tensor([0.4], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    )
    product_high_low_forward = product_policy.apply(
        torch,
        torch.tensor([0.4], dtype=torch.float32),
        torch.tensor([0.9], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    )
    product_high_low_reverse = product_policy.apply(
        torch,
        torch.tensor([-0.4], dtype=torch.float32),
        torch.tensor([0.9], dtype=torch.float32),
        torch.tensor([0.2], dtype=torch.float32),
    )
    check_true(
        "continuous_ranker_product_safety_weight_is_symmetric_and_concentrates_low_low_pairs",
        torch.allclose(product_low_low, torch.tensor([0.04]))
        and torch.allclose(product_high_low_forward, torch.tensor([0.18]))
        and torch.allclose(product_high_low_reverse, torch.tensor([0.18]))
        and float(product_low_low.item()) < 0.2,
    )

    margins = torch.tensor([0.4, 0.1, 0.7, -0.2], dtype=torch.float32)
    pure_mfe = torch.tensor([0.8, 0.4, 1.0, 0.2], dtype=torch.float32)
    dates_for_pair_weight = np.asarray(["2021-01-04"] * 4)
    all_context_one = torch.column_stack([pure_mfe, torch.ones(4, dtype=torch.float32)])
    unweighted_loss, unweighted_count = _pairwise_logistic_loss(
        torch,
        margins,
        pure_mfe,
        dates_for_pair_weight,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    for policy_id in (
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MFE_WINNER_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_CONFLICT_UNSAFE_WINNER_PREDICTED_SAFETY,
    ):
        weighted_loss, weighted_count = _pairwise_logistic_loss(
            torch,
            margins,
            all_context_one,
            dates_for_pair_weight,
            reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            pair_weight_policy=policy_id,
        )
        check_true(
            f"continuous_ranker_pair_weight_{policy_id}_collapses_to_base_when_context_is_one",
            weighted_count == unweighted_count
            and torch.allclose(
                weighted_loss.detach(), unweighted_loss.detach(), atol=1e-7, rtol=1e-7
            ),
        )

    synthetic_policy_id = "synthetic_third_pair_weight_policy"

    def _synthetic_right_context_multiplier(torch_module, _diff, _left, right):
        return right

    synthetic_policy = ContinuousRankerPairWeightPolicy(
        policy_id=synthetic_policy_id,
        context_source=CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
        compatible_reductions=(CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,),
        multiplier=_synthetic_right_context_multiplier,
        contract_pair_safety_weight="synthetic_right_context",
        contract_pair_weight_combination="delta_ndcg_times_synthetic_right_context",
        report_extension_title="Synthetic Third Pair Weight",
        report_first_note="- synthetic first note",
        report_second_note="- synthetic second note",
    )
    with patch.dict(
        continuous_ranker_runtime._CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY,
        {synthetic_policy_id: synthetic_policy},
        clear=False,
    ):
        normalized_reduction, resolved_synthetic = (
            normalize_continuous_ranker_pair_weight_configuration(
                CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
                synthetic_policy_id,
            )
        )
        synthetic_objective = ContinuousRankerObjectivePolicy(
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            pair_weight_policy=synthetic_policy_id,
            pair_target_schema=CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT,
        )
        synthetic_contract = breakout_quality_config.get_predicted_safety_pair_weight_contract(
            synthetic_policy_id
        )
        synthetic_loss, synthetic_count = _pairwise_logistic_loss(
            torch,
            margins,
            torch.column_stack([
                pure_mfe,
                torch.tensor([0.9, 0.8, 0.3, 0.2], dtype=torch.float32),
            ]),
            dates_for_pair_weight,
            reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            pair_weight_policy=synthetic_policy_id,
        )
        synthetic_report_extension = _pair_weight_report_extension(
            CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            synthetic_policy_id,
        )
        check_true(
            "continuous_ranker_synthetic_third_pair_weight_needs_only_registry_plugin",
            normalized_reduction
            == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
            and resolved_synthetic is synthetic_policy
            and synthetic_objective.pair_weight_policy == synthetic_policy_id
            and synthetic_contract.get("pair_safety_weight") == "synthetic_right_context"
            and synthetic_contract.get("pair_weight_combination")
            == "delta_ndcg_times_synthetic_right_context"
            and synthetic_loss is not None
            and synthetic_count > 0
            and synthetic_report_extension
            == (
                "Synthetic Third Pair Weight",
                "- synthetic first note",
                "- synthetic second note",
            ),
        )

    registered_pair_weight_ids = tuple(
        sorted(
            policy_id
            for policy_id in continuous_ranker_runtime._CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_REGISTRY
            if policy_id != CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE
        )
    )
    pair_weight_identity_leaks = []
    for source_root_name in ("filters", "services"):
        for source_path in sorted((project_root / source_root_name).rglob("*.py")):
            source_text = read_source_text(source_path)
            leaked_policy_ids = [
                policy_id for policy_id in registered_pair_weight_ids if policy_id in source_text
            ]
            if leaked_policy_ids:
                pair_weight_identity_leaks.append(
                    f"{source_path.relative_to(project_root).as_posix()}:"
                    + ",".join(leaked_policy_ids)
                )
    check(
        "continuous_ranker_generic_consumers_do_not_recognize_specific_pair_weight_ids",
        [],
        pair_weight_identity_leaks,
        note=(
            "all registered pair-weight identities, including future policies, must remain "
            "owned by config.breakout_quality_runtime; filters/services consume resolved "
            "policy contracts and must not branch on policy IDs"
        ),
    )

    registered_runtime_failures = []
    for registered_profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
        registered_profile = get_breakout_quality_experiment_profile(registered_profile_name)
        registered_spec = get_continuous_ranker_research_spec(registered_profile_name)
        recipe = get_continuous_ranker_execution_recipe(registered_profile_name)
        reasons = []

        expected_direct_fields = {
            "profile_name": registered_profile_name,
            "trainer_family": registered_spec.trainer_family,
            "training_objective": registered_profile.training_objective,
            "continuous_target_id": str(registered_profile.continuous_target_id),
            "loss_name": registered_profile.loss_name,
            "model_architecture": registered_profile.model_architecture,
            "training_label_scope": registered_profile.training_label_scope,
            "training_sample_scope": registered_profile.training_sample_scope,
            "score_semantic_id": registered_spec.score_semantic_id,
            "historical_pit_authorized": bool(registered_spec.selection_pit_authorized),
            "current_time_validation_authorized": bool(registered_spec.current_time_validation_authorized),
        }
        for field_name, expected_value in expected_direct_fields.items():
            actual_value = getattr(recipe, field_name)
            if actual_value != expected_value:
                reasons.append(f"{field_name}={actual_value!r}, expected={expected_value!r}")

        if recipe.dependency_spec.context_source != recipe.context_policy.source:
            reasons.append("dependency context source differs from runtime context source")
        if (
            recipe.context_policy.source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
            and CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE not in recipe.context_policy.roles
        ):
            reasons.append("persistent context source lacks coverage role")
        if not set(recipe.target_policy.context_roles).issubset(set(recipe.context_policy.roles)):
            reasons.append("target context roles are not preserved by runtime context policy")
        if (
            recipe.target_policy.context_source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
            and recipe.target_policy.context_source != recipe.context_policy.source
        ):
            reasons.append("target context source differs from runtime context source")

        weighted_pairwise = (
            recipe.objective_policy.pair_weight_policy
            != CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE
        )
        if weighted_pairwise != (
            recipe.objective_policy.pair_target_schema
            == CONTINUOUS_RANKER_PAIR_TARGET_SCHEMA_SCALAR_WITH_CONTEXT_WEIGHT
        ):
            reasons.append("pair-weight policy and pair-target schema are inconsistent")
        if (
            weighted_pairwise
            and CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT not in recipe.context_policy.roles
        ):
            reasons.append("weighted pairwise objective lacks pair-weight context role")
        if bool(recipe.training_policy.uses_pairwise_loss) != (recipe.pairwise_reduction is not None):
            reasons.append("training pairwise capability differs from objective reduction")
        try:
            recipe.training_policy.validate_profile_loss_metric(
                loss_name=registered_profile.loss_name,
                epoch_selection_metric=registered_profile.epoch_selection_metric,
            )
            recipe.training_policy.validate_research_spec(
                pairwise_reduction=registered_spec.pairwise_reduction,
                pair_weight_policy=registered_spec.pair_weight_policy,
                secondary_pair_scope=registered_spec.secondary_pair_scope,
                secondary_pair_scope_threshold=registered_spec.secondary_pair_scope_threshold,
            )
        except Exception as exc:
            reasons.append(
                f"training composition validation failed: {type(exc).__name__}: {exc}"
            )
        expected_pairwise_kind = (
            CONTINUOUS_RANKER_PAIRWISE_KIND_NONE
            if recipe.pairwise_reduction is None
            else CONTINUOUS_RANKER_PAIRWISE_KIND_PARETO
            if recipe.pairwise_reduction
            == continuous_ranker_runtime.CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE
            else CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR
        )
        if recipe.training_policy.pairwise_kind != expected_pairwise_kind:
            reasons.append(
                "training composition pairwise kind differs from resolved objective reduction"
            )
        if (
            registered_spec.pair_weight_policy is not None
            and not recipe.training_policy.allows_context_pair_weight
        ):
            reasons.append("research spec pair weight bypasses training composition capability")
        if (
            registered_spec.secondary_pair_scope
            not in recipe.training_policy.allowed_secondary_pair_scopes
        ):
            reasons.append("research spec secondary scope bypasses training composition capability")
        if not recipe.training_policy.profile_loss_metrics:
            reasons.append("registered profile lacks canonical loss/epoch metric composition")

        expected_dependencies = [ARTIFACT_DATASET_CORE]
        if recipe.dependency_spec.requires_continuous_target_artifact:
            expected_dependencies.append(ARTIFACT_CONTINUOUS_TARGET)
        context_artifact_spec = maybe_predicted_context_artifact_spec(
            recipe.dependency_spec.context_source
        )
        if context_artifact_spec is not None:
            expected_dependencies.append(context_artifact_spec.artifact_type)
        actual_dependencies = list(required_upstream_artifact_types(registered_profile_name))
        if actual_dependencies != expected_dependencies:
            reasons.append(
                f"dependencies={actual_dependencies!r}, expected={expected_dependencies!r}"
            )

        try:
            registered_training_semantics = training_semantics(registered_profile)

            def _semantic_values(node, key):
                values = []
                if isinstance(node, dict):
                    for child_key, child_value in node.items():
                        if child_key == key:
                            values.append(child_value)
                        values.extend(_semantic_values(child_value, key))
                elif isinstance(node, (list, tuple)):
                    for child_value in node:
                        values.extend(_semantic_values(child_value, key))
                return values

            expected_head_weighting = (
                recipe.training_policy.artifact_head_weighting_semantic()
            )
            actual_head_weightings = _semantic_values(
                registered_training_semantics, "head_weighting"
            )
            if expected_head_weighting is None:
                if actual_head_weightings:
                    reasons.append(
                        "single-head composition unexpectedly persists artifact head_weighting"
                    )
            elif actual_head_weightings != [expected_head_weighting]:
                reasons.append(
                    "artifact head_weighting differs from canonical training composition: "
                    f"actual={actual_head_weightings!r}, expected={[expected_head_weighting]!r}"
                )

            if weighted_pairwise:
                expected_pair_weight_contract = (
                    breakout_quality_config.get_predicted_safety_pair_weight_contract(
                        recipe.objective_policy.pair_weight_policy
                    )
                )
                actual_pair_weight_contract = dict(
                    (registered_training_semantics.get("pairwise_contract") or {}).get(
                        "predicted_safety_pair_weight_contract"
                    )
                    or {}
                )
                if actual_pair_weight_contract != expected_pair_weight_contract:
                    reasons.append(
                        "training semantics pair-weight contract differs from runtime registry"
                    )
        except Exception as exc:
            reasons.append(f"training_semantics failed: {type(exc).__name__}: {exc}")

        if reasons:
            registered_runtime_failures.append(
                f"{registered_profile_name}: " + "; ".join(reasons)
            )

    check(
        "continuous_ranker_all_registered_profiles_share_generic_runtime_contract",
        [],
        registered_runtime_failures,
    )
    check(
        "continuous_ranker_generic_contract_covers_every_registered_profile",
        len(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES),
        len(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES) - len(registered_runtime_failures),
    )

    config_source = read_source_text("config/breakout_quality.py")
    trainer_source = read_source_text(
        "services/breakout_quality/train_continuous_ranker.py"
    )
    training_contract_source = read_source_text(
        "filters/breakout_quality/ranker_training_contract.py"
    )
    check_true(
        "continuous_ranker_profile_and_research_spec_delegate_objective_legality_to_composition_owner",
        "expected_loss = {" not in config_source
        and "scoped secondary pair supervision只允許支援該capability的training objective"
        not in config_source
        and ".validate_profile_loss_metric(" in config_source
        and ".validate_research_spec(" in config_source,
        note=(
            "profile/research declarations must consume the runtime composition registry "
            "instead of maintaining objective-name capability matrices"
        ),
    )
    check_true(
        "continuous_ranker_artifact_head_weighting_derives_from_composition_owner",
        ".artifact_head_weighting_semantic()" in training_contract_source
        and "get_continuous_ranker_primary_pair_weight_policy(recipe.training_objective)"
        not in training_contract_source,
        note=(
            "persisted head weighting and truth-side primary weighting must consume the "
            "same ContinuousRankerTrainingPolicy composition as the trainer"
        ),
    )

    check_true(
        "continuous_ranker_trainer_consumes_supervision_modes_and_head_combination_from_composition",
        "training_policy.primary_supervision_mode" in trainer_source
        and "training_policy.secondary_supervision_mode" in trainer_source
        and "training_policy.primary_pair_weight_policy" in trainer_source
        and "training_policy.head_loss_combination" in trainer_source
        and "training_policy.head_loss_component_count" in trainer_source
        and "get_continuous_ranker_primary_pair_weight_policy(training_objective)"
        not in trainer_source,
        note=(
            "generic trainer may dispatch reusable primitive handlers, but scientific "
            "supervision composition/weighting must come from ContinuousRankerTrainingPolicy"
        ),
    )

    raw = np.asarray([1.0, 3.0, 2.0, 5.0, 5.0, 9.0], dtype=np.float32)
    valid = np.ones((6,), dtype=bool)
    dates = pd.Series(["2020-01-02"] * 3 + ["2021-05-03"] * 3)
    percentiles = build_daily_percentile_targets(raw, valid, dates)
    check(
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
    check(
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
    check(
        "continuous_ranker_singleton_date_uses_neutral_half_percentile",
        (0.5,),
        tuple(float(value) for value in singleton_percentile),
    )

    changed_oos = raw.copy()
    changed_oos[3:] = np.asarray([-100.0, 500.0, 0.0], dtype=np.float32)
    changed_percentiles = build_daily_percentile_targets(changed_oos, valid, dates)
    check(
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
        check(
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
        check_true("continuous_ranker_rejects_target_from_different_dataset_artifacts", stale_dataset_rejected)

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
        check_true("continuous_ranker_rejects_tampered_target_artifact", tamper_rejected)

    app_path = Path(__file__).resolve().parents[2] / "services" / "research" / "breakout_quality_application.py"
    tree = read_source_ast(app_path)
    command_modules = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    check(
        "continuous_ranker_research_command_is_registered",
        "services.breakout_quality.ranker_cli",
        command_modules.get("train-continuous-ranker"),
    )

    project_root = Path(__file__).resolve().parents[2]
    ranker_source = (
        project_root
        / "services"
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
    from config.breakout_quality import (
        BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
        BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
    )
    from services.breakout_quality.train_continuous_ranker import (
        _iter_materialized_feature_batches,
    )
    synthetic_feature_bank = np.arange(96, dtype=np.float32).reshape(16, 2, 3)
    synthetic_batches = [
        np.asarray([5, 1, 9], dtype=np.int64),
        np.asarray([2, 7], dtype=np.int64),
        np.asarray([11, 12, 13, 14], dtype=np.int64),
    ]
    serial_batches = list(
        _iter_materialized_feature_batches(
            synthetic_feature_bank, synthetic_batches,
            prefetch_batches=0, prefetch_workers=1,
        )
    )
    prefetched_batches = list(
        _iter_materialized_feature_batches(
            synthetic_feature_bank, synthetic_batches,
            prefetch_batches=BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES,
            prefetch_workers=BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS,
        )
    )
    check_true(
        "continuous_ranker_multiworker_prefetch_preserves_exact_batch_identity_and_values",
        BREAKOUT_QUALITY_CONTINUOUS_RANKER_TRAIN_PREFETCH_BATCHES == 8
                and BREAKOUT_QUALITY_CONTINUOUS_RANKER_PREFETCH_WORKERS == 4
                and len(serial_batches) == len(prefetched_batches)
                and all(np.array_equal(a[0], b[0]) for a, b in zip(serial_batches, prefetched_batches))
                and all(np.array_equal(a[1], b[1]) for a, b in zip(serial_batches, prefetched_batches))
                and "torch.cuda.Stream" in ranker_source
                and "non_blocking=True" in ranker_source
                and ".pin_memory()" in ranker_source,
    )

    check(
        "continuous_ranker_reuses_active_two_logit_head_and_pass_probability",
        (True, True, True),
        (
                    "self.classifier = nn.Linear(classifier_input, 2)" in inception_source,
                    "torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]" in ranker_source,
                    '"model_state_dict"' in ranker_source and "torch.save(" in ranker_source,
                ),
    )
    train_source = (
        project_root
        / "services"
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
        / "services" / "breakout_quality" / "export_scores.py"
    ).read_text(encoding="utf-8")
    check(
        "continuous_ranker_is_cli_only_and_does_not_pollute_interactive_menu",
        (True, True, True, True),
        (
                    command_modules.get("train-continuous-ranker")
                    == "services.breakout_quality.ranker_cli",
                    'print("[10] 11B 同日 Percentile Ranker（research-only）")' not in app_source,
                    'elif choice == "10":' not in app_source,
                    "_interactive_train_continuous_ranker" not in app_source,
                ),
    )

    from services.research import breakout_quality_application as research_app

    model_gate_settings = SimpleNamespace(
        is_binary_classification=False,
        is_continuous_ranker=True,
        rolling_authorized=False,
        filter_id="synthetic_quality",
        model_architecture="synthetic_arch",
        experiment_profile="synthetic_model_gate_profile",
        training_objective=TRAINING_OBJECTIVE_DAILY_RAW_R_REGRESSION,
        seed=42,
    )
    ready_plan = SimpleNamespace(blocked=False, actions=(), overall_status="READY")
    gate_commands = []

    def _record_model_gate_command(command, args, *, program_name, **_kwargs):
        gate_commands.append((str(command), list(args), str(program_name)))
        return 23

    with (
        patch.object(research_app, "_print_workflow_status"),
        patch.object(
            research_app, "get_continuous_ranker_research_spec",
            return_value=SimpleNamespace(model_research_id="SYNTHETIC-MODEL"),
        ),
        patch.object(
            research_app, "_load_reusable_continuous_forward_contract",
            return_value=(None, "synthetic missing"),
        ),
        patch.object(research_app, "_collect_continuous_research_input_plan", return_value=ready_plan),
        patch.object(research_app, "_render_continuous_research_input_plan"),
        patch.object(research_app, "_prompt_bool", return_value=True),
        patch.object(research_app, "_prepare_continuous_research_inputs", return_value=0),
        patch.object(research_app, "_run_command", side_effect=_record_model_gate_command),
    ):
        model_gate_rc = research_app._run_continuous_forward_model_gate(
            "apps/research.py model", model_gate_settings
        )
    check_true(
        "model_gate_only_continuous_profile_uses_direct_forward_model_gate_without_pit",
        model_gate_rc == 23
        and gate_commands == [(
            "train-continuous-ranker",
            [
                "--filter-id", "synthetic_quality",
                "--model-architecture", "synthetic_arch",
                "--experiment-profile", "synthetic_model_gate_profile",
                "--seed", "42",
            ],
            "apps/research.py model",
        )],
    )

    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
        patch.object(
            research_app,
            "get_continuous_ranker_research_spec",
            return_value=SimpleNamespace(
                reference_profile_name=None, model_research_id="SYNTHETIC-MODEL"
            ),
        ),
        patch("builtins.input", return_value=""),
        patch.object(research_app, "_run_continuous_forward_model_gate", return_value=29) as forward_gate,
        patch.object(research_app, "_interactive_continuous_pit_validation", return_value=31) as pit_gate,
    ):
        menu_rc = research_app._interactive_model_research("apps/research.py model")
    check_true(
        "model_gate_only_continuous_profile_enter_routes_to_forward_gate_not_rolling_pit",
        menu_rc == 29
        and forward_gate.call_count == 1
        and pit_gate.call_count == 0,
    )

    # Regression for B338: even if the current Training Model object carries a false
    # rolling flag (synthetic historical state), [4] belongs to the shared compare list
    # and must not be re-gated by that current Training Model.
    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
        patch.object(
            research_app,
            "get_continuous_ranker_research_spec",
            return_value=SimpleNamespace(
                reference_profile_name=None, model_research_id="SYNTHETIC-MODEL"
            ),
        ),
        patch("builtins.input", side_effect=["4", "0"]),
        patch.object(research_app, "_run_configured_model_comparison", return_value=0) as rolling_compare,
    ):
        rolling_compare_menu_rc = research_app._interactive_model_research("apps/research.py model")
    check_true(
        "rolling_comparison_menu_ignores_current_training_model_veto_and_delegates_to_shared_list",
        rolling_compare_menu_rc == 0
        and rolling_compare.call_count == 1
        and bool(rolling_compare.call_args.kwargs.get("rolling")) is True,
    )

    app_source = (project_root / "services" / "research" / "breakout_quality_application.py").read_text(encoding="utf-8")
    check_true(
        "continuous_model_menu_uses_shared_forward_rolling_train_compare_robustness_workflow",
        'render_menu_item(1, "Forward OOS 模型訓練", default=True)' in app_source
        and 'render_menu_item(2, "Rolling OOS 模型訓練")' in app_source
        and 'render_menu_item(3, "Forward OOS 模型比較")' in app_source
        and 'render_menu_item(4, "Rolling OOS 模型比較")' in app_source
        and 'render_menu_item(5, "Forward OOS Robustness 模型測試")' in app_source
        and 'render_menu_item(6, "Rolling OOS Robustness 模型測試")' in app_source
        and 'render_menu_item(7, "Timing Mode｜Rolling 訓練前後比較  [工程]")' in app_source
        and 'render_menu_item(3, "Fixed-Window Rolling")' not in app_source
        and "查看目前Workflow、工件與模型報表" not in app_source
        and "Actual MFE×Safety Truth Geometry（只讀）" not in app_source,
    )
    daily_ranker_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    pit_score_source = (
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    check_true(
        "model_workflow_ui_uses_mr_identity_one_status_table_and_inline_daily_target_progress",
        'print("Training Model：" + _model_identity_text(_model_display_id(settings.experiment_profile)))' in app_source
        and 'print(f"Training Profile：{settings.experiment_profile}")' not in app_source
        and 'def _model_identity_text(value: object, *, target: str = "console")' in app_source
        and 'paint(text, "cyan"' in app_source
        and 'def _print_model_action_status(rows)' in app_source
        and app_source.count('_print_model_action_status(') >= 5
        and 'paint("工件狀態", "cyan"' in app_source
        and 'paint("Workflow 狀態", "cyan"' not in app_source
        and 'model/report identity READY' not in app_source
        and ' Standard SOP | {mode} | ' not in app_source
        and 'from core.display_common import InlineProgress' in daily_ranker_source
        and 'daily_target_progress.update(' in daily_ranker_source
        and 'daily_target_progress.finish()' in daily_ranker_source
        and 'print("[Daily target/index] 開始建立daily stock-day index與40D target...' not in daily_ranker_source,
    )
    check_true(
        "model_comparison_direct_run_palette_seed_progress_and_pit_inline_refresh",
        'return paint(str(label), "light_yellow", enabled=color, bold=True)' in app_source
        and 'return f"### {markdown_tone(label, \'light_yellow\', bold=True)}"' in app_source
        and 'f"seed={int(seed)} ({int(index)}/{int(total)})"' in app_source
        and 'for seed_index, (seed, payload, path, reason) in enumerate(seed_states, start=1):' in app_source
        and '確認產生{mode_label}多模型Standard SOP比較；完整者REUSE，缺失者由canonical producer補建' not in app_source
        and '確認產生{mode_label} Robustness多模型Standard SOP比較；缺失seed由canonical producer補建' not in app_source
        and 'daily_data_progress = InlineProgress()' in pit_score_source
        and 'daily_data_progress.update(' in pit_score_source
        and 'daily_data_progress.finish()' in pit_score_source
        and 'print("[PIT data] 建立daily stock-day index／40D target...' not in pit_score_source,
    )
    check_true(
        "forward_checkpoint_reuse_is_offered_to_standard_and_robustness_rolling_only_through_fitting_identity_cache_bridge",
        'def _load_canonical_forward_checkpoint_import_source(settings):' in app_source
        and app_source.count('_append_pit_forward_checkpoint_import_args(') >= 3
        and 'forward_payload, forward_report_path, _forward_reason = _load_forward_robustness_seed(' in app_source
        and '"--checkpoint-cache-root", str(BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT)' in app_source
        and '"--checkpoint-import-model-dir", str(Path(model_dir).resolve())' in app_source
        and '"--checkpoint-import-report-path", str(Path(report_path).resolve())' in app_source
        and 'def _forward_checkpoint_import_issues(' in pit_score_source
        and 'execution_plan=plan' in pit_score_source
        and 'checkpoint_sample_scope = str(' in pit_score_source
        and 'checkpoint selected_epoch mismatch' not in pit_score_source,
    )

    rolling_settings = SimpleNamespace(
        rolling_authorized=True,
        filter_id="synthetic_quality",
        model_architecture="synthetic_arch",
        experiment_profile="synthetic_rolling_profile",
    )
    forward_checkpoint_source = (Path("/synthetic/forward/model"), Path("/synthetic/forward/report.json"))
    with (
        patch.object(research_app, "_print_workflow_status"),
        patch.object(
            research_app, "_load_reusable_rolling_standard_report",
            return_value=(None, None, "synthetic Rolling PIT missing"),
        ),
        patch.object(
            research_app, "_load_canonical_forward_checkpoint_import_source",
            return_value=(forward_checkpoint_source, None),
        ),
        patch.object(research_app, "_print_model_action_status") as rolling_status,
        patch.object(research_app, "_collect_continuous_research_input_plan", return_value=ready_plan),
        patch.object(research_app, "_render_continuous_research_input_plan"),
        patch.object(
            research_app, "get_continuous_ranker_research_spec",
            return_value=SimpleNamespace(model_research_id="SYNTHETIC-ROLLING"),
        ),
        patch.object(research_app, "_run_continuous_pit_profile", return_value=0) as pit_profile,
    ):
        rolling_rc = research_app._run_continuous_rolling_mode_direct(
            "apps/research.py model", rolling_settings, prompt_for_build=False
        )
    rolling_reason = str(rolling_status.call_args.args[0][0][3])
    check_true(
        "standard_rolling_detects_validated_forward_model_report_and_passes_checkpoint_candidate_to_pit_producer",
        rolling_rc == 0
        and pit_profile.call_count == 1
        and pit_profile.call_args.kwargs.get("forward_checkpoint_source") == forward_checkpoint_source
        and "Forward model/report READY" in rolling_reason
        and "exact fitting identity" in rolling_reason,
    )

    synthetic_build_args = ["--filter-id", "synthetic_quality"]
    research_app._append_pit_forward_checkpoint_import_args(
        synthetic_build_args, forward_checkpoint_source
    )
    check_true(
        "forward_checkpoint_import_bridge_passes_model_and_report_as_one_atomic_argument_pair",
        synthetic_build_args[-4:] == [
            "--checkpoint-import-model-dir", str(forward_checkpoint_source[0].resolve()),
            "--checkpoint-import-report-path", str(forward_checkpoint_source[1].resolve()),
        ],
    )

    check_true(
        "model_robustness_restart_reports_ready_seed_reuse_and_completed_seed_done",
        'seed_text = _seed_progress_text(seed, seed_index, len(seeds))' in app_source
        and 'styled_workflow_status("[REUSE]")' in app_source
        and 'styled_workflow_status("[DONE]")' in app_source
        and 'f" | {mode_label} Standard SOP READY"' in app_source,
    )


    shared_training = breakout_quality_config.get_breakout_quality_model_research_settings()
    shared_test = breakout_quality_config.get_breakout_quality_model_test_settings().model_profiles
    canonical_training_pair = tuple(
        str(value).strip()
        for value in breakout_quality_config.BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    reference_test_profiles = tuple(
        (str(model_id).strip(), str(profile_name).strip())
        for model_id, profile_name
        in breakout_quality_config.BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES
    )
    config_source = (project_root / "config" / "breakout_quality.py").read_text(encoding="utf-8")
    check_true(
        "model_compare_test_list_structurally_contains_training_model_and_reference_controls",
        len(canonical_training_pair) == 2
        and canonical_training_pair[1] == str(shared_training.experiment_profile)
        and canonical_training_pair in shared_test
        and all(item in shared_test for item in reference_test_profiles)
        and "BREAKOUT_QUALITY_MODEL_TEST_PROFILES = _merge_breakout_quality_model_profiles(" in config_source
        and "(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,)" in config_source,
    )
    synthetic_new_training_pair = ("MR-SYNTHETIC-NEW", "synthetic_new_training_profile")
    synthetic_merged = breakout_quality_config._merge_breakout_quality_model_profiles(
        reference_test_profiles,
        (synthetic_new_training_pair,),
    )
    check_true(
        "future_training_model_is_automatically_injected_into_compare_and_robustness_membership",
        synthetic_new_training_pair in synthetic_merged
        and all(item in synthetic_merged for item in reference_test_profiles),
    )
    configured_workflow_profiles = breakout_quality_config.get_breakout_quality_model_workflow_profile_names()
    expected_workflow_profiles = tuple(dict.fromkeys(
        [str(shared_training.experiment_profile)]
        + [str(profile_name) for _model_id, profile_name in shared_test]
    ))
    check_true(
        "model_workflow_membership_is_the_only_current_mode_authorization_selector",
        tuple(configured_workflow_profiles) == expected_workflow_profiles
        and shared_training.rolling_authorized
        and shared_training.robustness_authorized
        and all(
            breakout_quality_config.get_breakout_quality_workflow_settings(
                experiment_profile=str(profile_name)
            ).rolling_authorized
            and breakout_quality_config.get_breakout_quality_workflow_settings(
                experiment_profile=str(profile_name)
            ).robustness_authorized
            for _model_id, profile_name in shared_test
        )
        and "current_model_workflow_rolling_authorized" not in config_source
        and "current_model_workflow_robustness_authorized" not in config_source,
    )
    check_true(
        "comparison_and_robustness_menu_use_shared_list_without_current_training_model_regate",
        "Rolling比較尚未授權" not in app_source
        and "尚未通過單一Seed Gate；Robustness尚未授權" not in app_source
        and "Current Model尚未授權Rolling Robustness" not in app_source
        and '_run_configured_model_comparison(program_name, rolling=True)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=False)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=True)' in app_source,
    )

    from filters.breakout_quality.contract import RUNTIME_SCOPE_WORKFLOW
    from services.breakout_quality.export_scores import _run_daily_continuous_workflow_export
    from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data

    # Regression guard for the Round-2 execution-recipe migration: the daily loader returns
    # the canonical profile object in ContinuousRankerDataBundle, so ``profile`` must be a
    # bound local derived from the recipe rather than an accidental unresolved global name.
    check_true(
        "daily_ranker_loader_binds_bundle_profile_from_canonical_runtime_recipe",
        "profile" in load_daily_universal_ranker_data.__code__.co_varnames,
    )

    event_scope_runtime_export_rejected = False
    try:
        _run_daily_continuous_workflow_export(
            root=project_root,
            args=SimpleNamespace(scope=RUNTIME_SCOPE_WORKFLOW),
            profile=profile,
        )
    except ValueError as exc:
        event_scope_runtime_export_rejected = (
            "daily_eligible_stock_days" in str(exc)
        )

    check(
        "event_scope_continuous_ranker_stays_out_of_binary_runtime_while_daily_runtime_export_is_separate",
        (True, True, True, True),
        (
                    "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in train_source,
                    "research-only continuous ranker artifact不得載入正式binary runtime contract" in artifact_source,
                    "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in app_source,
                    event_scope_runtime_export_rejected,
                ),
    )

    check(
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

    check(
        "pass_conditional_ranker_is_cli_only_and_uses_existing_command",
        (True, True, True, True),
        (
                    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
                    in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
                    get_continuous_ranker_research_spec(
                        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
                    ).trainer_family
                    == CONTINUOUS_RANKER_TRAINER_EVENT,
                    "11G" not in app_source[app_source.index("def _interactive_model_research"):app_source.index("def run_model_training_menu")],
                    command_modules.get("train-continuous-ranker")
                    == "services.breakout_quality.ranker_cli",
                ),
    )

    ranker_api_source = (
        project_root / "services" / "breakout_quality" / "ranker_training.py"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        project_root / "services" / "breakout_quality" / "continuous_ranker_pipeline.py"
    ).read_text(encoding="utf-8")
    daily_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    public_api_names = (
        "select_epoch",
        "fit_final",
        "predict_scores",
        "split_metrics",
        "training_semantics",
        "resolve_training_output_paths",
        "daily_rank_metrics",
        "daily_top_k_metrics",
        "calculate_spearman",
    )
    check_true(
        "ranker_training_public_api_is_single_shared_boundary_without_private_cross_module_calls",
        all(name in ranker_api_source for name in public_api_names)
                and "ranker_training as ranker_api" in pipeline_source
                and "ranker_training as ranker_api" in daily_source
                and "ranker_impl._" not in pipeline_source
                and "ranker_impl._" not in daily_source
                and "ranker_impl=" not in daily_source
                and "ranker_impl=sys.modules" not in ranker_source,
    )

    from services.breakout_quality import ranker_training as ranker_training_api
    from services.breakout_quality import train_continuous_ranker as canonical_ranker
    check_true(
        "continuous_ranker_run_does_not_rebind_shared_public_api_names",
        not (set(public_api_names) & set(canonical_ranker.run.__code__.co_varnames)),
    )
    check_true(
        "legacy_private_ranker_names_alias_the_public_training_api_without_second_implementation",
        canonical_ranker._select_epoch is ranker_training_api.select_epoch
                and canonical_ranker._fit_final is ranker_training_api.fit_final
                and canonical_ranker._predict_scores is ranker_training_api.predict_scores
                and canonical_ranker._split_metrics is ranker_training_api.split_metrics
                and canonical_ranker._training_semantics is ranker_training_api.training_semantics
                and canonical_ranker._training_output_paths is ranker_training_api.resolve_training_output_paths
                and canonical_ranker._daily_rank_metrics is ranker_training_api.daily_rank_metrics
                and canonical_ranker._daily_top_k_metrics is ranker_training_api.daily_top_k_metrics
                and canonical_ranker._spearman is ranker_training_api.calculate_spearman,
    )

    summary["profile"] = STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    summary["pass_conditional_profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["training_objective"] = profile.training_objective
    return results, summary

def validate_breakout_quality_all_event_no_time_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_ALL_EVENT_NO_TIME_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from tools.filters.breakout_quality.train_continuous_ranker import (
        _profile_contract,
        _scope_group_ids,
        build_daily_percentile_targets,
        parse_args as parse_continuous_ranker_args,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    )
    check(
        "all_event_no_time_profile_uses_existing_target_and_all_label_scope",
        (
                    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
                    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
                    TRAINING_LABEL_SCOPE_ALL,
                    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
                    "mse",
                    "mean_daily_spearman",
                    False,
                ),
        (
                    profile.training_objective,
                    profile.continuous_target_id,
                    profile.training_label_scope,
                    profile.training_sample_scope,
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
    check("all_event_scope_keeps_pass_and_reject_groups", (0, 1, 2), tuple(int(value) for value in ids))

    raw = np.asarray([1.0, 9.0, 5.0], dtype=np.float32)
    pct = build_daily_percentile_targets(
        raw, np.ones(3, dtype=bool), group_table["date"]
    )
    check(
        "all_event_daily_percentile_ranks_across_binary_labels",
        (0.0, 1.0, 0.5),
        tuple(float(value) for value in pct),
    )

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE]
    )
    contract = _profile_contract(profile)
    check(
        "mr12a_cli_profile_identity_is_explicit_and_research_only",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE, "12A", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    summary["training_label_scope"] = TRAINING_LABEL_SCOPE_ALL
    return results, summary

def validate_breakout_quality_pairwise_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PAIRWISE_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from filters.breakout_quality.models.factory import require_torch
    from config.strategy_compare import get_strategy_comparison_settings
    from tools.filters.breakout_quality.train_continuous_ranker import (
        PAIRWISE_TRAINING_CONTRACT,
        _daily_top_k_metrics,
        _date_coherent_batches,
        _pairwise_logistic_loss,
        _profile_contract,
        _training_semantics,
        _trade_alignment_metrics,
        parse_args as parse_continuous_ranker_args,
    )
    import config.breakout_quality as breakout_quality_config
    from config.breakout_quality import (
        get_breakout_quality_continuous_ranker_comparison_settings,
    )
    from filters.breakout_quality.continuous_ranker_quality import (
        exact_random_top_k_baseline,
    )
    from tools.filters.breakout_quality.compare_continuous_rankers import (
        _load_model_frame,
        _dynamic_orderable_frame,
        _evaluate_dynamic_k_strata,
        _evaluate_fixed_k_sweep,
        _evaluate_paired_frame,
        _evaluate_reference_subset_attribution,
        _evaluate_score_event_date_comparability,
        _render_dynamic_k_strata_table,
        _render_reference_subset_attribution_table,
        _render_score_event_date_comparability_table,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    check(
        "mr12b_profile_changes_only_learning_objective_with_same_target_scope_architecture_family",
        (
                    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
                    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
                    TRAINING_LABEL_SCOPE_ALL,
                    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
                    "pairwise_logistic",
                    "mean_daily_spearman",
                    False,
                ),
        (
                    profile.training_objective,
                    profile.continuous_target_id,
                    profile.training_label_scope,
                    profile.training_sample_scope,
                    profile.loss_name,
                    profile.epoch_selection_metric,
                    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
                    in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
                ),
    )

    semantics = _training_semantics(profile)
    check(
        "mr12b_pairwise_artifact_semantics_are_explicit_and_runtime_score_is_unchanged",
        (
                    "same_date_non_tied_target_pairs",
                    "equal_pair_weight",
                    "pass_logit_minus_reject_logit",
                    "whole_date_pack_no_date_split",
                    "softmax_pass_probability",
                ),
        tuple(
                    semantics["pairwise_contract"][key]
                    for key in ("pair_scope", "pair_weighting", "model_margin", "batching", "runtime_score")
                ),
    )
    assert semantics["pairwise_contract"] == PAIRWISE_TRAINING_CONTRACT

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE]
    )
    contract = _profile_contract(profile)
    check(
        "mr12b_cli_and_registry_identity_are_explicit",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE, "12B", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    dates = pd.Series(["2024-01-02"] * 3 + ["2024-01-03"] * 2 + ["2024-01-04"] * 3)
    batches = _date_coherent_batches(
        np.arange(8, dtype=np.int64), dates, batch_size=4, seed=42
    )
    batch_by_group = {
        int(group_id): int(batch_index)
        for batch_index, batch in enumerate(batches)
        for group_id in batch
    }
    date_batch_counts = []
    for _date, day in pd.DataFrame({"date": dates, "group_id": np.arange(8)}).groupby("date"):
        date_batch_counts.append(len({batch_by_group[int(group_id)] for group_id in day["group_id"]}))
    check("pairwise_training_batches_never_split_same_date_competition_set", (1, 1, 1), tuple(date_batch_counts))

    torch, _nn = require_torch()
    targets = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0], dtype=torch.float32)
    loss_dates = np.asarray(["2024-01-02"] * 3 + ["2024-01-03"] * 2)
    good_margin = torch.tensor([-2.0, 0.0, 2.0, -1.0, 1.0], dtype=torch.float32, requires_grad=True)
    bad_margin = torch.tensor([2.0, 0.0, -2.0, 1.0, -1.0], dtype=torch.float32, requires_grad=True)
    good_loss, good_pairs = _pairwise_logistic_loss(torch, good_margin, targets, loss_dates)
    bad_loss, bad_pairs = _pairwise_logistic_loss(torch, bad_margin, targets, loss_dates)
    good_loss.backward()
    check(
        "pairwise_logistic_prefers_correct_within_day_order_and_has_finite_gradient",
        (True, 4, 4, True),
        (
                    bool(float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())),
                    int(good_pairs),
                    int(bad_pairs),
                    bool(torch.isfinite(good_margin.grad).all().item()),
                ),
    )

    report_dates = np.asarray(["2024-02-01"] * 6 + ["2024-02-02"] * 6 + ["2024-02-05"] * 2)
    report_raw_target = np.asarray([6, 5, 4, 3, 2, 1, 12, 10, 8, 6, 4, 2, 100, -100], dtype=np.float64)
    report_percentile = np.concatenate([
        np.tile(np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64), 2),
        np.asarray([0.0, 1.0], dtype=np.float64),
    ])
    report_metrics = _daily_top_k_metrics(
        report_dates,
        report_raw_target / float(report_raw_target.max()),
        report_raw_target,
        report_percentile,
        top_k=3,
        boundary_width=2,
    )
    check(
        "continuous_ranker_top_k_report_measures_perfect_order_and_k_boundary",
        (1.0, 1.0, 1.0, 2, 2, 1, True),
        (
                    round(float(report_metrics["ndcg_at_k"]), 6),
                    round(float(report_metrics["oracle_top_k_overlap"]), 6),
                    round(float(report_metrics["boundary_concordance"]), 6),
                    int(report_metrics["competition_date_count"]),
                    int(report_metrics["boundary_date_count"]),
                    int(report_metrics["excluded_non_competition_date_count"]),
                    bool(float(report_metrics["boundary_raw_target_gap"]) > 0.0),
                ),
    )

    random_baseline = exact_random_top_k_baseline(
        np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64),
        np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
        top_k=3,
    )
    check(
        "continuous_ranker_random_baseline_is_exact_not_monte_carlo",
        (True, 0.0, 0.5, 0.5, 0.0),
        (
                    bool(0.0 < float(random_baseline["ndcg_at_k"]) < 1.0),
                    float(random_baseline["top_k_raw_target_lift"]),
                    float(random_baseline["oracle_top_k_overlap"]),
                    float(random_baseline["boundary_concordance"]),
                    float(random_baseline["boundary_raw_target_gap"]),
                ),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        score_path = root / "continuous_ranker_scores.csv"
        pd.DataFrame([
            {
                "ticker": "A", "date": "2021-01-04", "group_index": 0,
                "split": "oos", "target_raw_r": 0.2,
                "target_daily_percentile": 1.0, "model_score": 0.8,
            },
            {
                "ticker": "B", "date": "2021-01-04", "group_index": 1,
                "split": "oos", "target_raw_r": 0.1,
                "target_daily_percentile": 0.5, "model_score": 0.6,
            },
            {
                "ticker": "C", "date": "2021-12-31", "group_index": 2,
                "split": "oos", "target_raw_r": np.nan,
                "target_daily_percentile": np.nan, "model_score": 0.7,
            },
        ]).to_csv(score_path, index=False, encoding="utf-8-sig")
        fake_contract = SimpleNamespace(
            score_path=score_path,
            continuous_target_id=profile.continuous_target_id,
            report_path=root / "report.json",
            model_information_cutoff="2020-12-31",
            available_from="2021-01-04",
            available_through="2021-12-31",
        )
        with patch(
            "tools.filters.breakout_quality.compare_continuous_rankers.load_continuous_ranker_oos_contract",
            return_value=fake_contract,
        ):
            comparable_frame, _comparable_meta = _load_model_frame(
                root=root,
                filter_id="synthetic",
                architecture="synthetic",
                model_id="synthetic",
                profile=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
            )
    check(
        "continuous_ranker_quality_comparison_uses_target_evaluable_subset_not_runtime_score_universe",
        (["A", "B"], True),
        (
                    comparable_frame["ticker"].tolist(),
                    bool(np.isfinite(comparable_frame["target_raw_r"].to_numpy(dtype=np.float64)).all()),
                ),
    )

    with (
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES",
            (
                ("CFG-A", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE),
                ("CFG-B", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE),
                ("CFG-C", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE),
            ),
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM",
            "CFG-REFERENCE",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR",
            ("CFG-B", "CFG-A"),
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL",
            "Config comparison",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES",
            (1, 4, 9),
        ),
    ):
        configured_comparison = (
            get_breakout_quality_continuous_ranker_comparison_settings()
        )
    check(
        "continuous_ranker_p2_comparison_settings_follow_isolated_config_override",
        (
                    ("CFG-A", "CFG-B", "CFG-C"),
                    "CFG-REFERENCE",
                    ("CFG-B", "CFG-A"),
                    "Config comparison",
                    (1, 4, 9),
                ),
        (
                    configured_comparison.model_ids,
                    configured_comparison.reference_arm,
                    configured_comparison.summary_pair,
                    configured_comparison.menu_label,
                    configured_comparison.fixed_k_values,
                ),
    )

    paired_frame = pd.DataFrame(
        {
            "date": ["2024-03-01"] * 6,
            "ticker": [f"T{i}" for i in range(6)],
            "group_index": np.arange(6, dtype=np.int64),
            "split": ["oos"] * 6,
            "target_raw_r": np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
            "target_daily_percentile": np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64),
            "score__MR-12A": np.asarray([6, 4, 5, 3, 2, 1], dtype=np.float64),
            "score__MR-12B": np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
            "score__MR-12C": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
        }
    )
    paired_eval = _evaluate_paired_frame(
        paired_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        k_by_date={"2024-03-01": 3},
        boundary_width=2,
    )
    ba = paired_eval["paired_contrasts"]["MR-12B_minus_MR-12A"]["metrics"]
    cb = paired_eval["paired_contrasts"]["MR-12C_minus_MR-12B"]["metrics"]
    check(
        "continuous_ranker_p2_uses_same_day_paired_deltas",
        (1, True, True, 1.0),
        (
                    int(paired_eval["competition_date_count"]),
                    bool(float(ba["ndcg_at_k"]["mean_delta"]) > 0.0),
                    bool(float(cb["ndcg_at_k"]["mean_delta"]) < 0.0),
                    float(paired_eval["models"]["MR-12B"]["boundary_concordance"]),
                ),
    )

    fixed_prefix = _evaluate_fixed_k_sweep(
        paired_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        k_values=(1, 2, 3),
        boundary_width=2,
    )
    check(
        "continuous_ranker_p2_fixed_k_prefix_sweep_reuses_same_oos_candidate_universe",
        (("1", "2", "3"), 1, True, True),
        (
                    tuple(sorted(fixed_prefix, key=int)),
                    int(fixed_prefix["1"]["competition_date_count"]),
                    bool(
                        float(
                            fixed_prefix["1"]["paired_contrasts"]["MR-12B_minus_MR-12A"]["metrics"]["ndcg_at_k"]["mean_delta"]
                        ) >= 0.0
                    ),
                    bool(
                        float(
                            fixed_prefix["3"]["paired_contrasts"]["MR-12C_minus_MR-12B"]["metrics"]["ndcg_at_k"]["mean_delta"]
                        ) < 0.0
                    ),
                ),
    )

    with tempfile.TemporaryDirectory() as td:
        pair_dir = Path(td)
        pd.DataFrame(
            [
                {
                    "Date": "2024-03-04",
                    "Resource_Aware_Max_DL_Eligible": True,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
                {
                    "Date": "2024-03-05",
                    "Resource_Aware_Max_DL_Eligible": True,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
                {
                    "Date": "2024-03-06",
                    "Resource_Aware_Max_DL_Eligible": False,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
            ]
        ).to_csv(pair_dir / "score_ranking_daily_capacity.csv", index=False)
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "breakout_quality_score_date": "2024-03-01",
                    "breakout_quality_score": 0.75,
                    "breakout_quality_score_unavailable_reason": "",
                }
                for trade_date, signal_date in (
                    ("2024-03-04", "2024-03-01"),
                    ("2024-03-05", "2024-03-02"),
                )
                for ticker in ("A", "B", "C", "D")
            ]
        ).to_csv(pair_dir / "score_ranking_orderable_candidates.csv", index=False)
        dynamic_models = {}
        target_raw = [4.0, 3.0, 2.0, 1.0]
        target_pct = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        for model_id, scores in {
            "MR-12A": [0.8, 0.7, 0.6, 0.5],
            "MR-12B": [0.9, 0.8, 0.7, 0.6],
            "MR-12C": [0.6, 0.7, 0.8, 0.9],
        }.items():
            dynamic_models[model_id] = pd.DataFrame(
                {
                    "ticker": ["A", "B", "C", "D"],
                    "date": ["2024-03-01"] * 4,
                    "group_index": [1, 2, 3, 4],
                    "split": ["oos"] * 4,
                    "target_raw_r": target_raw,
                    "target_daily_percentile": target_pct,
                    "model_score": scores,
                }
            )
        dynamic_frame, dynamic_coverage = _dynamic_orderable_frame(
            pair_dir=pair_dir,
            model_frames=dynamic_models,
            model_ids=("MR-12A", "MR-12B", "MR-12C"),
        )
    dynamic_strata = _evaluate_dynamic_k_strata(
        dynamic_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        boundary_width=2,
    )
    check(
        "continuous_ranker_dynamic_k_uses_runtime_score_event_date_for_later_occurrences",
        (8, (2,), 2, 1.0, 8, 4, 1.0, 1.0, ("2",), 2),
        (
                    int(len(dynamic_frame)),
                    tuple(sorted(set(int(value) for value in dynamic_frame["dynamic_k"]))),
                    int(dynamic_coverage["full_score_coverage_date_count"]),
                    float(dynamic_coverage["full_score_coverage_rate"]),
                    int(dynamic_coverage["runtime_score_event_date_row_count"]),
                    int(dynamic_coverage["score_event_date_differs_from_signal_date_row_count"]),
                    float(dynamic_coverage["common_complete_candidate_row_rate"]),
                    float(dynamic_coverage["reference_runtime_scored_candidate_row_rate"]),
                    tuple(sorted(dynamic_strata)),
                    int(dynamic_strata["2"]["competition_date_count"]),
                ),
    )
    dynamic_strata_table = _render_dynamic_k_strata_table(
        dynamic_strata,
        summary_pair=("MR-12B", "MR-12A"),
    )
    check(
        "continuous_ranker_dynamic_k_strata_console_renderer_uses_shared_table_contract",
        (True, True, True),
        (
                    bool(dynamic_strata_table.strip()),
                    "MR-12B NDCG" in dynamic_strata_table,
                    "Boundary Δ" in dynamic_strata_table,
                ),
    )
    reference_subset = _evaluate_reference_subset_attribution(
        dynamic_frame,
        dynamic_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        boundary_width=2,
    )
    reference_subset_table = _render_reference_subset_attribution_table(
        reference_subset,
        summary_pair=("MR-12B", "MR-12A"),
    )
    reference_k2 = dict(reference_subset.get("2") or {})
    reference_event_k2 = dict(reference_k2.get("event_universe") or {})
    reference_orderable_k2 = dict(reference_k2.get("orderable_universe") or {})
    check(
        "continuous_ranker_reference_subset_attribution_holds_dates_and_k_constant_across_universes",
        (("2",), 2, 2, 2, True, True),
        (
                    tuple(sorted(reference_subset)),
                    int(reference_k2.get("reference_common_complete_date_count", 0)),
                    int(reference_event_k2.get("competition_date_count", 0)),
                    int(reference_orderable_k2.get("competition_date_count", 0)),
                    bool(reference_subset_table.strip()),
                    "Orderable NDCG Δ" in reference_subset_table,
                ),
    )

    score_date_frame = pd.DataFrame(
        {
            "date": ["2024-03-10"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "score_event_date": ["2024-03-10", "2024-03-10", "2024-03-01", "2024-03-01"],
            "dynamic_k": [1, 1, 1, 1],
            "target_raw_r": [4.0, 3.0, 2.0, 1.0],
            "score__MR-12A": [0.9, 0.8, 0.7, 0.6],
            "score__MR-12B": [0.2, 0.1, 0.9, 0.8],
            "score__MR-12C": [0.9, 0.8, 0.7, 0.6],
        }
    )
    score_date_diag = _evaluate_score_event_date_comparability(
        score_date_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
    )
    score_date_table = _render_score_event_date_comparability_table(
        score_date_diag,
        summary_pair=("MR-12B", "MR-12A"),
    )
    same_scope = dict(
        (score_date_diag.get("pair_scopes") or {}).get("same_score_event_date_pairs") or {}
    )
    cross_scope = dict(
        (score_date_diag.get("pair_scopes") or {}).get("cross_score_event_date_pairs") or {}
    )
    same_delta = dict(
        (same_scope.get("contrasts") or {}).get("MR-12B_minus_MR-12A") or {}
    ).get("pairwise_concordance_delta")
    cross_delta = dict(
        (cross_scope.get("contrasts") or {}).get("MR-12B_minus_MR-12A") or {}
    ).get("pairwise_concordance_delta")
    check(
        "continuous_ranker_orderable_score_date_comparability_separates_within_date_from_cross_date_pairs",
        (0.5, 1, 1.0, 1.0, 0.0, -1.0, True, True),
        (
                    round(float(score_date_diag.get("carried_candidate_row_rate")), 6),
                    int(score_date_diag.get("mixed_score_event_date_count", 0)),
                    float(score_date_diag.get("mixed_score_event_date_rate")),
                    float(score_date_diag.get("k1_mixed_score_event_date_rate")),
                    round(float(same_delta), 6),
                    round(float(cross_delta), 6),
                    "Cross score-date" in score_date_table,
                    "K=1 cross score-date" in score_date_table,
                ),
    )

    from filters.breakout_quality.workflow_io import PROJECT_ROOT as breakout_project_root
    missing_trade = _trade_alignment_metrics(
        pd.DataFrame(),
        breakout_project_root / "outputs" / "filters" / "breakout_quality" / "synthetic_missing_trade",
    )
    check(
        "continuous_ranker_missing_trade_reason_uses_project_relative_display_path",
        (False, True, False),
        (
                    bool(missing_trade.get("available")),
                    str(missing_trade.get("reason", "")).startswith("not found: outputs/"),
                    str(breakout_project_root).replace("\\", "/") in str(missing_trade.get("reason", "")),
                ),
    )

    strategy = get_strategy_comparison_settings()
    historical_arm_contract = (
        ("C17", "CONT12A", "resource-aware-continuous-max-dl"),
        ("C18", "CONT12A", "resource-aware-continuous-max-dl-feasible-ascent"),
        ("C19", "CONT12B", "resource-aware-continuous-max-dl"),
        ("C20", "CONT12B", "resource-aware-continuous-max-dl-feasible-ascent"),
    )
    check(
        "mr12a_and_mr12b_historical_c17_c18_selector_matrix_remains_registered",
        historical_arm_contract,
        tuple(
                    (
                        arm_id,
                        strategy.arms.get(arm_id).dl_id if strategy.arms.get(arm_id) is not None else None,
                        strategy.arms.get(arm_id).dl_runtime_mode if strategy.arms.get(arm_id) is not None else None,
                    )
                    for arm_id, _dl_id, _runtime_mode in historical_arm_contract
                ),
    )
    required_contrasts = {
        "C19-C17": ("C19", "C17"),
        "C20-C18": ("C20", "C18"),
        "C20-C19": ("C20", "C19"),
    }
    check(
        "mr12b_historical_strategy_matrix_is_registered_independent_of_active_config",
        required_contrasts,
        {
                    contrast_id: (strategy.contrasts[contrast_id].left, strategy.contrasts[contrast_id].right)
                    for contrast_id in required_contrasts
                    if contrast_id in strategy.contrasts
                },
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    summary["training_objective"] = TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING
    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_daily_full_list_ndcg_pairwise_contract_case(_base_params):
    """Pin MR-13E as a full-list position-aware Delta-NDCG RankNet experiment."""

    case_id = "BREAKOUT_QUALITY_DAILY_FULL_LIST_NDCG_PAIRWISE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
    )
    from filters.breakout_quality.models.factory import require_torch
    from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
    from tools.filters.breakout_quality.train_continuous_ranker import (
        parse_args as parse_continuous_ranker_args,
    )

    profile_a = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    profile_e = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec_a = get_continuous_ranker_research_spec(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    spec_e = get_continuous_ranker_research_spec(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )

    fixed_fields = (
        "optimizer_name",
        "training_sampling_mode",
        "training_objective",
        "continuous_target_id",
        "loss_name",
        "epoch_selection_metric",
        "training_label_scope",
        "training_sample_scope",
    )
    check(
        "mr13e_changes_only_pairwise_reduction_while_daily_profile_contract_stays_fixed",
        tuple(getattr(profile_a, field) for field in fixed_fields),
        tuple(getattr(profile_e, field) for field in fixed_fields),
    )
    check(
        "mr13e_research_identity_and_full_list_delta_ndcg_reduction_are_explicit",
        (
                    "MR-13E",
                    spec_a.trainer_family,
                    spec_a.score_semantic_id,
                    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
                ),
        (
                    spec_e.model_research_id,
                    spec_e.trainer_family,
                    spec_e.score_semantic_id,
                    spec_e.pairwise_reduction,
                ),
    )
    from services.breakout_quality.train_continuous_ranker import training_semantics
    check(
        "mr13e_artifact_semantics_disclose_full_list_delta_ndcg_weighting",
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        training_semantics(profile_e)["pairwise_contract"]["pair_weighting"],
    )

    from filters.breakout_quality import ranking_score_store as ranking_store

    with tempfile.TemporaryDirectory() as td:
        contract_root = Path(td)
        model_path = contract_root / "model.pt"
        manifest_path = contract_root / "manifest.json"
        report_path = contract_root / ranking_store.CONTINUOUS_RANKER_REPORT_FILENAME
        score_path = contract_root / ranking_store.DAILY_RANKER_OOS_SCORE_FILENAME
        model_path.write_bytes(b"synthetic-model")
        score_path.write_bytes(b"synthetic-score")

        canonical_semantics = training_semantics(profile_e)
        score_eligibility = ranking_store.build_score_eligibility_contract(profile_e)
        coverage = {
            "inference_eligible_groups": 1,
            "target_evaluable_groups": 1,
            "future_target_required_for_score": False,
        }
        manifest_payload = {
            "filter_id": "synthetic",
            "model_architecture": "synthetic_arch",
            "experiment_profile": profile_e.name,
            "training_objective": profile_e.training_objective,
            "training_label_scope": profile_e.training_label_scope,
            "training_sample_scope": profile_e.training_sample_scope,
            "training_semantics": canonical_semantics,
            "continuous_target_id": profile_e.continuous_target_id,
            "score_eligibility_contract": score_eligibility,
            "forward_score_coverage": coverage,
            "model": {},
            "research_outputs": {"oos_scores_gzip": {}},
            "outer_oos_policy": {
                "oos_start_date": "2021-01-01",
                "configured_oos_end_date": "2021-12-31",
            },
            "model_information_cutoff": "2020-12-31",
        }
        report_payload = {
            "filter_id": "synthetic",
            "model_architecture": "synthetic_arch",
            "experiment_profile": profile_e.name,
            "training": {
                "objective": profile_e.training_objective,
                "loss": profile_e.loss_name,
                "sample_scope": profile_e.training_sample_scope,
                "training_label_scope": profile_e.training_label_scope,
                "batching": canonical_semantics["batching"],
                "pairwise_contract": canonical_semantics["pairwise_contract"],
                "seed": 42,
            },
            "experiment_settings": profile_e.as_manifest_payload(),
            "score_eligibility_contract": score_eligibility,
            "forward_score_coverage": coverage,
            "status": "RESULT_AVAILABLE",
            "artifacts": {"oos_scores_gzip": {}},
        }
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        report_path.write_text(json.dumps(report_payload), encoding="utf-8")

        fake_score_table = pd.DataFrame(
            [{
                "ticker": "SYN",
                "date": "2021-01-04",
                "group_index": 0,
                "model_score": 0.9,
            }]
        )
        fake_score_table.attrs["available_from"] = "2021-01-04"
        fake_score_table.attrs["available_through"] = "2021-01-04"

        def _load_contract():
            ranking_store.load_continuous_ranker_oos_contract.cache_clear()
            with (
                patch.object(
                    ranking_store,
                    "resolve_filter_artifact_paths",
                    return_value=SimpleNamespace(
                        manifest_path=manifest_path,
                        model_path=model_path,
                    ),
                ),
                patch.object(
                    ranking_store,
                    "resolve_filter_model_output_dir",
                    return_value=contract_root,
                ),
                patch.object(
                    ranking_store,
                    "resolve_continuous_ranker_oos_score_path",
                    return_value=score_path,
                ),
                patch.object(
                    ranking_store,
                    "_validate_file_record_simple",
                    return_value=None,
                ),
                patch.object(
                    ranking_store,
                    "load_continuous_ranker_oos_score_table",
                    return_value=fake_score_table,
                ),
            ):
                return ranking_store.load_continuous_ranker_oos_contract(
                    str(contract_root),
                    "synthetic",
                    "synthetic_arch",
                    profile_e.name,
                )

        canonical_runtime_contract_passed = False
        try:
            loaded_contract = _load_contract()
            canonical_runtime_contract_passed = (
                loaded_contract.experiment_profile == profile_e.name
                and loaded_contract.seed == 42
            )
        except Exception as exc:
            canonical_runtime_contract_passed = False
            summary["runtime_oos_contract_error"] = f"{type(exc).__name__}: {exc}"

        # Frozen MR-13E was trained before raw-R regression metadata existed.
        # A later non-applicable null key must not invalidate that checkpoint.
        historical_semantics = json.loads(json.dumps(canonical_semantics))
        historical_semantics.pop("raw_r_regression_contract", None)
        manifest_payload["training_semantics"] = historical_semantics
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        historical_schema_contract_passed = False
        try:
            historical_loaded = _load_contract()
            historical_schema_contract_passed = (
                historical_loaded.experiment_profile == profile_e.name
            )
        except Exception as exc:
            summary["historical_semantics_error"] = f"{type(exc).__name__}: {exc}"

        wrong_semantics = json.loads(json.dumps(canonical_semantics))
        wrong_semantics["pairwise_contract"]["pair_weighting"] = (
            CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR
        )
        manifest_payload["training_semantics"] = wrong_semantics
        report_payload["training"]["pairwise_contract"] = wrong_semantics[
            "pairwise_contract"
        ]
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        report_path.write_text(json.dumps(report_payload), encoding="utf-8")
        wrong_weighting_rejected = False
        try:
            _load_contract()
        except ValueError:
            wrong_weighting_rejected = True

    check(
        "runtime_oos_contract_uses_same_profile_driven_pairwise_semantics_as_trainer",
        (True, True, True),
        (
                    canonical_runtime_contract_passed,
                    historical_schema_contract_passed,
                    wrong_weighting_rejected,
                ),
    )
    args = parse_continuous_ranker_args(
        ["--experiment-profile", DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE]
    )
    check(
        "mr13e_is_available_through_existing_profile_driven_cli",
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        args.experiment_profile,
    )
    torch, _nn = require_torch()
    import torch.nn.functional as F

    targets = torch.tensor([1.0, 0.8, 0.2, 0.0], dtype=torch.float32)
    margins = torch.tensor([1.6, 0.4, 1.0, -0.2], dtype=torch.float32, requires_grad=True)
    dates = np.asarray(["2024-01-02"] * 4)
    actual_loss, actual_count = _pairwise_logistic_loss(
        torch,
        margins,
        targets,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )

    margin_diff = margins[:, None] - margins[None, :]
    target_diff = targets[:, None] - targets[None, :]
    upper = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
    comparable = upper & (target_diff != 0)
    pair_losses = F.softplus(
        -torch.sign(target_diff[comparable]) * margin_diff[comparable]
    )
    positions = torch.arange(1, 5, dtype=torch.float32)
    discounts = 1.0 / torch.log2(positions + 1.0)
    predicted_order = torch.argsort(margins.detach(), descending=True, stable=True)
    discount_by_item = torch.empty_like(discounts)
    discount_by_item[predicted_order] = discounts
    ideal_order = torch.argsort(targets, descending=True, stable=True)
    idcg = (targets[ideal_order] * discounts).sum()
    weight_matrix = (
        torch.abs(target_diff.detach())
        * torch.abs(discount_by_item[:, None] - discount_by_item[None, :])
        / idcg
    )
    expected_weights = weight_matrix[comparable]
    expected_loss = (pair_losses * expected_weights).sum() / expected_weights.sum()
    actual_loss.backward()
    check(
        "full_list_delta_ndcg_loss_matches_raw_percentile_full_rank_swap_weighting",
        (round(float(expected_loss.detach().cpu().item()), 12), 6, True),
        (
                    round(float(actual_loss.detach().cpu().item()), 12),
                    int(actual_count),
                    bool(torch.isfinite(margins.grad).all().item()),
                ),
        tol=1e-10,
    )

    correct_margins = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    predicted_order = torch.argsort(correct_margins, descending=True, stable=True)
    discount_by_item = torch.empty_like(discounts)
    discount_by_item[predicted_order] = discounts
    correct_weight_matrix = (
        torch.abs(targets[:, None] - targets[None, :])
        * torch.abs(discount_by_item[:, None] - discount_by_item[None, :])
        / idcg
    )
    top_adjacent_weight = float(correct_weight_matrix[0, 1].item())
    bottom_adjacent_weight = float(correct_weight_matrix[2, 3].item())
    check_true(
        "full_list_delta_ndcg_naturally_prioritizes_top_positions_without_k_boundary_or_lambda",
        bool(top_adjacent_weight > bottom_adjacent_weight > 0.0),
    )

    summary["profile"] = DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    summary["model_research_id"] = spec_e.model_research_id
    return results, summary




def validate_breakout_quality_full_horizon_target_components_contract_case(_base_params):
    """Protect reusable full-horizon opportunity and pure-MFE target mathematics."""
    case_id = "BREAKOUT_QUALITY_FULL_HORIZON_TARGET_COMPONENTS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    horizon = int(spec.horizon_bars)
    highs = np.full(horizon, 102.0, dtype=np.float64)
    lows = np.full(horizon, 99.0, dtype=np.float64)
    highs[4], lows[4], highs[19], lows[19] = 103.0, 89.0, 125.0, 95.0
    truncated = daily_opportunity_no_time_target_from_cached_path(
        highs, lows, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    full = daily_full_horizon_opportunity_target_from_cached_path(
        highs, lows, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "full_horizon_keeps_breach_diagnostic_without_truncating_later_peak",
        (5, 20, 1.4, True),
        (int(full.first_risk_breach_bar), int(full.opportunity_bar), round(float(full.target_raw_r), 6), truncated.opportunity_bar < full.first_risk_breach_bar),
    )

    no_breach_high = np.linspace(101.0, 112.0, horizon, dtype=np.float64)
    no_breach_low = np.full(horizon, 97.0, dtype=np.float64)
    old_no_breach = daily_opportunity_no_time_target_from_cached_path(
        no_breach_high, no_breach_low, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    new_no_breach = daily_full_horizon_opportunity_target_from_cached_path(
        no_breach_high, no_breach_low, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "full_horizon_no_breach_path_matches_reference_target",
        (old_no_breach.target_raw_r, old_no_breach.opportunity_bar, old_no_breach.adverse_return_to_peak),
        (new_no_breach.target_raw_r, new_no_breach.opportunity_bar, new_no_breach.adverse_return_to_peak),
        tol=1e-12,
    )

    dates = pd.date_range("2020-01-01", periods=horizon + 3, freq="D")
    frame = pd.DataFrame({
        "Open": np.full(len(dates), 100.0),
        "High": np.concatenate(([100.0], highs, [102.0, 102.0])),
        "Low": np.concatenate(([100.0], lows, [99.0, 99.0])),
        "Close": np.full(len(dates), 100.0),
        "Volume": np.full(len(dates), 1000.0),
    }, index=dates)
    full_batch = compute_daily_opportunity_target_batch(
        frame, np.asarray([0], dtype=np.int64), spec=spec, target_id=DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    )
    pure = daily_full_horizon_pure_mfe_target_from_cached_path(
        highs, lows, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    pure_batch = compute_daily_opportunity_target_batch(
        frame, np.asarray([0], dtype=np.int64), spec=spec, target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    check(
        "full_horizon_targets_preserve_scalar_vector_and_removed_adverse_relation",
        (
            round(float(full.target_raw_r), 6), round(float(pure.target_raw_r), 6),
            full.opportunity_bar, full.first_risk_breach_bar,
            round(float(full.adverse_return_to_peak) / float(spec.risk_budget_return), 6),
        ),
        (
            round(float(full_batch.target_raw_r[0]), 6), round(float(pure_batch.target_raw_r[0]), 6),
            pure.opportunity_bar, pure.first_risk_breach_bar,
            round(float(pure.target_raw_r - full.target_raw_r), 6),
        ),
    )

    opportunity_contract = build_daily_full_horizon_opportunity_contract(DEFAULT_LABEL_POLICY)
    pure_contract = build_daily_full_horizon_pure_mfe_contract(DEFAULT_LABEL_POLICY)
    check(
        "full_horizon_target_contracts_disclose_no_truncation_and_pure_mfe_semantics",
        (DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID, True, DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID, False, False),
        (
            opportunity_contract["target_id"],
            "diagnostic only" in str(opportunity_contract["risk_rule"]).lower() and "never truncates" in str(opportunity_contract["risk_rule"]).lower(),
            pure_contract["target_id"], bool(pure_contract["adverse_penalty_included"]),
            "adverse_return" in str(pure_contract["formula"]),
        ),
    )

    from services.breakout_quality.daily_target_comparison import _comparison_metrics, _pure_mfe_float32_relation_tolerance
    audit_frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-02"]),
        "reference_target_r": [0.2, 0.5, 0.1, 0.9], "candidate_target_r": [0.2, 1.2, 0.1, 1.1],
        "first_risk_breach_bar": [-1, 5, -1, 8], "reference_opportunity_bar": [4, 3, 6, 7],
        "candidate_opportunity_bar": [4, 20, 6, 15], "minimum_low_return": [-0.095, -0.105, -0.05, -0.12],
        "reference_adverse_return": [0.02, 0.10, 0.03, 0.10], "candidate_adverse_return": [0.02, 0.10, 0.03, 0.10],
    })
    metrics, _daily = _comparison_metrics(audit_frame, top_k=1, risk_barrier_return=-0.10, barrier_band_return=0.01)
    check(
        "full_horizon_target_audit_keeps_breach_change_and_no_breach_invariant",
        (0.5, 0.5, 0.0),
        (float(metrics["risk_breach_rate"]), float(metrics["changed_target_rate"]), float(metrics["no_breach_max_abs_target_delta_r"])),
        tol=1e-12,
    )
    persisted_candidate = np.asarray([500.0], dtype=np.float32).astype(np.float64)
    persisted_reference = np.asarray([499.9], dtype=np.float32).astype(np.float64)
    persisted_adverse = np.asarray([0.01], dtype=np.float32).astype(np.float64)
    persisted_error = np.abs((persisted_candidate - persisted_reference) - persisted_adverse / float(spec.risk_budget_return))
    tolerance = _pure_mfe_float32_relation_tolerance(
        persisted_candidate, persisted_reference, persisted_adverse, risk_budget_return=float(spec.risk_budget_return),
    )
    check("pure_mfe_relation_tolerance_tracks_float32_quantization", (True, True), (bool(persisted_error[0] > 2e-6), bool(persisted_error[0] <= tolerance[0])))

    from filters.breakout_quality.daily_ranker_data import resolve_daily_training_universe_start
    benchmark_dates = pd.date_range("2004-01-01", periods=305, freq="D")
    check(
        "daily_training_universe_start_uses_complete_feature_window",
        pd.Timestamp(benchmark_dates[299]).normalize(),
        resolve_daily_training_universe_start(benchmark_dates, feature_window_bars=300),
    )
    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_reusable_model_component_contract_case(_base_params):
    """Protect reusable ranker components without pinning closed experiment identities."""

    case_id = "BREAKOUT_QUALITY_REUSABLE_MODEL_COMPONENTS"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
        CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R,
        SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from config.strategy_compare import (
        get_strategy_comparison_settings,
        get_strategy_rolling_test_modes,
    )
    from core.params_io import params_to_json_dict
    from core.strategy_params import V16StrategyParams
    from filters.breakout_quality.continuous_target import (
        DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID,
        build_daily_full_horizon_equal_rank_mfe_low_adverse_contract,
    )
    from filters.breakout_quality.daily_ranker_data import build_equal_rank_mfe_low_adverse_target
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.architectures import (
        ARCHITECTURE_DESCRIPTORS,
        ArchitectureDescriptor,
        get_architecture_descriptor,
    )
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.runtime_registry import registered_runtime_builder_keys
    from filters.breakout_quality.models.spec import (
        ACTIVE_MODEL_ARCHITECTURES,
        LEGACY_MODEL_ARCHITECTURES,
        SUPPORTED_MODEL_ARCHITECTURES,
        INCEPTION_TIME_RISK_CONTEXT_V1,
        INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1,
        INCEPTION_TIME_SHARED_SAFETY_MFE_V1,
        INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1,
        INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1,
        INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1,
        PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1,
        INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1,
        INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1,
        INCEPTION_TIME_V1,
        get_model_spec,
        model_spec_from_manifest,
    )
    from filters.breakout_quality.models.spec_registry import MODEL_SPEC_BUILDERS
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from filters.breakout_quality.risk_normalized_target import (
        RISK_GEOMETRY_CONTEXT_FEATURES,
        RiskParamPeriod,
        _params_for_period,
        build_risk_target_contract,
        compute_risk_geometry,
        load_min_roos_risk_schedule,
        risk_normalized_target_from_future_path,
    )
    from services.breakout_quality import ranker_training as ranker_api
    from services.breakout_quality import train_continuous_ranker as training_module

    # Dual-component regression is a reusable learning formulation. Resolve the
    # registered capability owner dynamically instead of inventing a fake experiment
    # identity, so the generic engine remains strict about canonical profiles.
    dual_profile_name = next(
        profile_name
        for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if (
            get_continuous_ranker_execution_recipe(profile_name)
            .training_policy.semantics_contract_key
            == CONTINUOUS_RANKER_SEMANTICS_DUAL_COMPONENT_R
        )
    )
    dual_profile = get_breakout_quality_experiment_profile(dual_profile_name)
    dual_contract = dict(
        training_semantics(dual_profile).get("dual_component_r_regression_contract") or {}
    )
    check(
        "dual_component_regression_has_equal_primary_components_and_fixed_runtime_score",
        (
            "mean_mse_over_two_primary_r_components",
            "equal_by_mean_reduction_no_lambda",
            "predicted_favorable_mfe_r_minus_predicted_adverse_to_peak_r",
        ),
        (
            dual_contract.get("loss"),
            dual_contract.get("component_weighting"),
            dual_contract.get("runtime_score"),
        ),
    )
    group_table = pd.DataFrame(
        {
            "target_adverse_r": [0.25, 1.10, 0.0],
            "target_favorable_r": [1.75, 3.40, 0.80],
        }
    )
    raw_target = np.asarray([1.50, 2.30, 0.80], dtype=np.float32)
    component_target = training_module._training_target_for_profile(
        dual_profile,
        raw_target,
        np.asarray([0.2, 0.8, 0.5], dtype=np.float32),
        group_table,
    )
    reconstructed = component_target[:, LABEL_PASS] - component_target[:, LABEL_REJECT]
    check(
        "dual_component_targets_reconstruct_composite_r_exactly",
        tuple(round(float(x), 6) for x in raw_target),
        tuple(round(float(x), 6) for x in reconstructed),
    )
    fake_logits = np.asarray([[0.4, 2.0], [1.2, 3.5]], dtype=np.float32)
    with patch.object(training_module, "strict_parallel_batched_logits", return_value=fake_logits):
        prediction = training_module.predict_dual_component_r(
            None,
            None,
            np.empty((2, 1, 1), dtype=np.float32),
            np.empty((2, 0), dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            batch_size=2,
            plan=None,
        )
    check(
        "dual_component_prediction_exposes_components_and_difference_score",
        ((2.0, 3.5), (0.4, 1.2), (1.6, 2.3)),
        (
            tuple(round(float(x), 6) for x in prediction["predicted_favorable_r"]),
            tuple(round(float(x), 6) for x in prediction["predicted_adverse_r"]),
            tuple(round(float(x), 6) for x in prediction["model_score"]),
        ),
    )

    # Equal-rank is retained as a generic target transform, not as a permanent MR recipe.
    dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02", "2020-01-03"])
    composite, mfe_pct, low_adv_pct = build_equal_rank_mfe_low_adverse_target(
        np.asarray([3.0, 2.0, 1.0, 4.0], dtype=np.float64),
        np.asarray([2.0, 0.5, 1.0, 0.2], dtype=np.float64),
        np.ones(4, dtype=bool),
        dates,
    )
    check(
        "equal_rank_target_uses_same_date_component_percentiles_and_fixed_half_weights",
        (
            (1.0, 0.5, 0.0, 0.5),
            (0.0, 1.0, 0.5, 0.5),
            (0.5, 0.75, 0.25, 0.5),
        ),
        (
            tuple(round(float(x), 6) for x in mfe_pct),
            tuple(round(float(x), 6) for x in low_adv_pct),
            tuple(round(float(x), 6) for x in composite),
        ),
    )
    equal_rank_contract = build_daily_full_horizon_equal_rank_mfe_low_adverse_contract(
        DEFAULT_LABEL_POLICY
    )
    check(
        "equal_rank_target_contract_has_no_weight_tuning_or_strategy_membership",
        (DAILY_FULL_HORIZON_EQUAL_RANK_MFE_LOW_ADVERSE_TARGET_ID, 0.5, 0.5, "none", False),
        (
            equal_rank_contract["target_id"],
            float(equal_rank_contract["component_weights"]["mfe_percentile"]),
            float(equal_rank_contract["component_weights"]["low_adverse_percentile"]),
            equal_rank_contract["weight_tuning"],
            bool(equal_rank_contract["requires_strategy_candidate_membership"]),
        ),
    )

    # Risk-normalized target/context remains reusable independently of the closed experiments
    # that first exercised it.
    risk_contract = build_risk_target_contract(
        horizon_bars=int(DEFAULT_LABEL_POLICY.label_horizon_bars),
        param_policy="base-finalist-best",
    )
    check(
        "risk_normalized_target_uses_only_initial_risk_fields_and_canonical_accounting",
        (["atr_len", "atr_times_init"], False, True, False),
        (
            list(risk_contract["risk_fields"]),
            "high_len" in risk_contract["risk_fields"],
            "canonical" in str(risk_contract["accounting"]),
            bool(risk_contract["strategy_exit_path_used"]),
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        param_payload = params_to_json_dict(V16StrategyParams())
        param_payload.update({"atr_len": 14, "atr_times_init": 2.0})
        strategy_settings = get_strategy_comparison_settings("selection_pit")
        for source_id, years in (("selection_min_roos", range(2014, 2021)), ("min_roos", range(2021, 2027))):
            source = strategy_settings.parameter_sources[source_id]
            path = temp_root / str(source.path_template).format(param_filename="roos_base_best.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "meta": {},
                        "summary": {},
                        "params_ensemble_by_effective_date": {
                            f"{year}-01-01": [{"params": dict(param_payload)}] for year in years
                        },
                    }
                ),
                encoding="utf-8",
            )
        risk_schedule = load_min_roos_risk_schedule(temp_root, param_policy="base-finalist-best")
    check(
        "risk_schedule_reuses_historical_and_current_min_parameter_sources_without_backfill",
        (13, "2014-01-01", "2026-01-01", 14, 2.0),
        (
            len(risk_schedule),
            risk_schedule[0].start_date.date().isoformat(),
            risk_schedule[-1].start_date.date().isoformat(),
            risk_schedule[0].atr_len,
            risk_schedule[0].atr_times_init,
        ),
    )
    period = RiskParamPeriod(
        start_date=pd.Timestamp("2020-01-01"),
        end_date=pd.Timestamp("2020-12-31"),
        atr_len=14,
        atr_times_init=2.0,
        source_id="synthetic",
        params_signature="synthetic",
    )
    geometry = compute_risk_geometry(
        ticker="2330",
        decision_date="2020-06-01",
        reference_price=100.0,
        atr=2.5,
        period=period,
    )
    check(
        "risk_geometry_context_is_five_dimensional_finite_and_bounded",
        (True, 5, True, True),
        (
            bool(geometry.valid),
            len(geometry.context),
            bool(all(math.isfinite(value) for value in geometry.context)),
            bool(0.0 < float(geometry.context[4]) <= 1.0),
        ),
    )
    params = _params_for_period(period)
    horizon = int(DEFAULT_LABEL_POLICY.label_horizon_bars)
    future_dates = pd.date_range("2020-06-02", periods=horizon, freq="B")
    safe_target, safe_valid, _ = risk_normalized_target_from_future_path(
        ticker="2330",
        decision_date="2020-06-01",
        reference_price=geometry.reference_price,
        stop_price=geometry.stop_price,
        qty=geometry.qty,
        planned_initial_risk_milli=geometry.planned_initial_risk_milli,
        future_high=np.full(horizon, 105.0, dtype=np.float64),
        future_low=np.full(horizon, 99.0, dtype=np.float64),
        future_dates=future_dates,
        params=params,
    )
    stop_target, stop_valid, stop_reason = risk_normalized_target_from_future_path(
        ticker="2330",
        decision_date="2020-06-01",
        reference_price=geometry.reference_price,
        stop_price=geometry.stop_price,
        qty=geometry.qty,
        planned_initial_risk_milli=geometry.planned_initial_risk_milli,
        future_high=np.full(horizon, 120.0, dtype=np.float64),
        future_low=np.full(horizon, 94.0, dtype=np.float64),
        future_dates=future_dates,
        params=params,
    )
    check(
        "risk_normalized_target_deducts_cost_and_prioritizes_same_bar_stop",
        (True, True, True, "stop_first"),
        (
            bool(safe_valid and safe_target < 0.8),
            bool(stop_valid),
            bool(math.isclose(float(stop_target), -1.0, rel_tol=0.0, abs_tol=1e-12)),
            stop_reason,
        ),
    )
    torch, _nn = require_torch()
    sequence_model = build_active_model(10, 0, architecture=INCEPTION_TIME_V1)
    context_model = build_active_model(
        10, len(RISK_GEOMETRY_CONTEXT_FEATURES), architecture=INCEPTION_TIME_RISK_CONTEXT_V1
    )
    x = torch.zeros((2, int(DEFAULT_LABEL_POLICY.feature_window_bars), 10), dtype=torch.float32)
    check(
        "risk_context_architecture_accepts_five_dimensional_context_without_changing_output_schema",
        ((2, 2), (2, 2)),
        (
            tuple(sequence_model(x, torch.empty((2, 0), dtype=torch.float32)).shape),
            tuple(
                context_model(
                    x,
                    torch.zeros((2, len(RISK_GEOMETRY_CONTEXT_FEATURES)), dtype=torch.float32),
                ).shape
            ),
        ),
    )

    # Pareto supervision is retained as a generic pairwise reduction. Trade-off/tie pairs
    # contribute no gradient, and only strict same-date dominance is scored.
    pareto_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02"] * 4),
            "target_favorable_r": [5.0, 3.0, 4.0, 2.0],
            "target_adverse_r": [0.2, 0.8, 1.5, 0.1],
        }
    )
    economic_target = np.asarray([4.8, 2.2, 2.5, 1.9], dtype=np.float32)
    pareto_components = ranker_api.build_pareto_component_percentile_targets(
        pareto_table, economic_target
    )
    check(
        "pareto_targets_are_same_date_mfe_and_low_adverse_percentiles",
        (
            (1.0, 0.333333, 0.666667, 0.0),
            (0.666667, 0.333333, 0.0, 1.0),
        ),
        (
            tuple(round(float(x), 6) for x in pareto_components[:, 0]),
            tuple(round(float(x), 6) for x in pareto_components[:, 1]),
        ),
    )
    correct = ranker_api.pareto_pair_concordance_metrics(
        np.arange(4, dtype=np.int64),
        pareto_table,
        economic_target,
        np.asarray([0.9, 0.2, 0.1, 0.99], dtype=np.float32),
    )
    reversed_result = ranker_api.pareto_pair_concordance_metrics(
        np.arange(4, dtype=np.int64),
        pareto_table,
        economic_target,
        np.asarray([0.0, 0.8, 0.7, 0.99], dtype=np.float32),
    )
    check(
        "pareto_concordance_counts_only_strict_dominance_pairs",
        (2, 6, round(2 / 6, 6), 1.0, 0.0),
        (
            int(correct["comparable_pair_count"]),
            int(correct["all_pair_count"]),
            round(float(correct["comparable_pair_rate"]), 6),
            round(float(correct["mean_daily_pareto_pair_concordance"]), 6),
            round(float(reversed_result["mean_daily_pareto_pair_concordance"]), 6),
        ),
    )
    margins = torch.as_tensor([0.9, 0.2, 0.1, 0.99], dtype=torch.float32)
    target_tensor = torch.as_tensor(pareto_components, dtype=torch.float32)
    pareto_loss, pair_count = training_module._pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        pareto_table["date"].to_numpy(),
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_PARETO_DOMINANCE,
    )
    check(
        "pareto_pairwise_reduction_excludes_tradeoff_and_tie_pairs_from_gradient",
        (2, True),
        (int(pair_count), bool(pareto_loss is not None and torch.isfinite(pareto_loss).item())),
    )


    # Predicted-context persistence is one reusable capability. Exercise the registry,
    # Stage-1 orchestration and artifact round-trip behavior once for every registered
    # source instead of repeating implementation-source assertions in MR-13AC/AD/AE.
    from filters.breakout_quality.daily_ranker_data import build_daily_ranker_split
    from filters.breakout_quality.paths import (
        SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
        SELECTION_POINT_IN_TIME_SCORE_FILENAME,
    )
    from filters.breakout_quality.predicted_context_artifact import (
        load_validated_predicted_context,
        predicted_context_artifact_specs,
    )
    from services.breakout_quality import predicted_context as predicted_context_service

    context_specs = predicted_context_artifact_specs()
    registered_context_sources = {
        get_continuous_ranker_execution_recipe(profile_name).dependency_spec.context_source
        for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if get_continuous_ranker_execution_recipe(profile_name).dependency_spec.context_source
        != "none"
    }
    check(
        "predicted_context_registry_covers_every_registered_persistent_context_source",
        registered_context_sources,
        {spec.source for spec in context_specs},
    )
    check_true(
        "predicted_context_registry_has_unique_artifact_and_builder_identities",
        len({spec.artifact_type for spec in context_specs}) == len(context_specs)
        and len({spec.builder_type for spec in context_specs}) == len(context_specs),
    )

    context_orchestration_errors = []
    context_owner_errors = []
    context_roundtrip_errors = []
    readiness_summary = {
        "source_data_date_range": {"end": "2021-12-31"},
        "policy": {"dataset": "synthetic"},
        "dataset_artifacts": {"identity": "synthetic"},
    }
    outer_policy = {
        "selection_end_date": "2020-12-31",
        "oos_start_date": "2021-01-01",
        "effective_oos_end_date": "2021-12-31",
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        stage1_calls = []

        def fake_build_cross_fitted_context_scores(**kwargs):
            stage1_calls.append(dict(kwargs))
            directory = Path(kwargs["point_in_time_dir_override"])
            directory.mkdir(parents=True, exist_ok=True)
            is_forward = bool(kwargs["single_score_block"])
            score_date = "2021-01-04" if is_forward else "2020-12-30"
            cutoff = "2020-12-31" if is_forward else "2020-11-30"
            fold_id = "forward_fixed" if is_forward else "selection_crossfit"
            pd.DataFrame(
                {
                    "ticker": ["0050", "00632R"],
                    "date": [score_date, score_date],
                    "breakout_quality_score": [0.2, 0.8],
                    "fold_id": [fold_id, fold_id],
                    "model_information_cutoff": [cutoff, cutoff],
                }
            ).to_csv(
                directory / SELECTION_POINT_IN_TIME_SCORE_FILENAME,
                index=False,
                encoding="utf-8-sig",
            )
            (directory / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).write_text(
                "{}\n", encoding="utf-8"
            )
            return directory / SELECTION_POINT_IN_TIME_SCORE_FILENAME

        with (
            patch.object(
                predicted_context_service,
                "collect_dataset_readiness",
                return_value=SimpleNamespace(ready=True, summary=readiness_summary),
            ),
            patch.object(
                predicted_context_service,
                "resolve_breakout_quality_outer_policy",
                return_value=outer_policy,
            ),
            patch.object(
                predicted_context_service,
                "get_breakout_quality_workflow_settings",
                return_value=SimpleNamespace(
                    point_in_time_inner_validation_months=24,
                    point_in_time_fold_months=12,
                ),
            ),
            patch.object(
                predicted_context_service,
                "build_cross_fitted_context_scores",
                side_effect=fake_build_cross_fitted_context_scores,
            ),
        ):
            for spec in context_specs:
                candidates = [
                    profile_name
                    for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
                    if get_continuous_ranker_execution_recipe(
                        profile_name
                    ).dependency_spec.context_source
                    == spec.source
                ]
                if not candidates:
                    context_orchestration_errors.append(f"{spec.source}:no_consumer")
                    continue
                consumer_profile_name = next(
                    (
                        name
                        for name in candidates
                        if spec.owner_profile is None or name != spec.owner_profile
                    ),
                    candidates[0],
                )
                consumer_profile = get_breakout_quality_experiment_profile(
                    consumer_profile_name
                )
                before = len(stage1_calls)
                predicted_context_service.build_registered_predicted_context(
                    spec.source,
                    project_root=temp_root,
                    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                    model_architecture=str(consumer_profile.model_architecture),
                    experiment_profile=consumer_profile.name,
                    dataset="full",
                    max_tickers=0,
                )
                calls = stage1_calls[before:]
                if len(calls) != 2:
                    context_orchestration_errors.append(
                        f"{spec.source}:calls={len(calls)}"
                    )
                else:
                    selection_call, forward_call = calls
                    expected_common = (
                        spec.stage1_profile,
                        spec.stage1_architecture,
                        int(spec.stage1_seed),
                    )
                    actual_common = (
                        str(selection_call.get("experiment_profile") or ""),
                        str(selection_call.get("model_architecture") or ""),
                        int(selection_call.get("seed", -1)),
                    )
                    if actual_common != expected_common:
                        context_orchestration_errors.append(
                            f"{spec.source}:stage1={actual_common!r}"
                        )
                    selection_dir = Path(selection_call["point_in_time_dir_override"])
                    forward_dir = Path(forward_call["point_in_time_dir_override"])
                    if (
                        selection_dir.name != "stage1_selection_crossfit"
                        or forward_dir.name != "stage1_forward_fixed"
                        or bool(selection_call["single_score_block"])
                        or not bool(forward_call["single_score_block"])
                        or selection_dir == forward_dir
                    ):
                        context_orchestration_errors.append(
                            f"{spec.source}:selection_forward_isolation"
                        )

                frame, manifest = load_validated_predicted_context(
                    spec.source,
                    temp_root,
                    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                    model_architecture=str(consumer_profile.model_architecture),
                    experiment_profile=consumer_profile.name,
                    expected_dataset_policy=readiness_summary["policy"],
                    expected_dataset_artifacts=readiness_summary["dataset_artifacts"],
                )
                expected_owner = (
                    spec.resolve_owner_architecture(str(consumer_profile.model_architecture)),
                    spec.resolve_owner_profile(consumer_profile.name),
                )
                actual_owner = (
                    str(manifest.get("model_architecture") or ""),
                    str(manifest.get("experiment_profile") or ""),
                )
                if actual_owner != expected_owner:
                    context_owner_errors.append(
                        f"{spec.source}:{actual_owner!r}!={expected_owner!r}"
                    )
                source_stage1 = dict(manifest.get("source_stage1") or {})
                if (
                    str(source_stage1.get("research_id") or "") != spec.stage1_research_id
                    or str(source_stage1.get("profile") or "") != spec.stage1_profile
                    or str(source_stage1.get("architecture") or "")
                    != spec.stage1_architecture
                    or int(source_stage1.get("seed", -1)) != int(spec.stage1_seed)
                ):
                    context_owner_errors.append(f"{spec.source}:stage1_manifest")
                if (
                    frame["ticker"].tolist()
                    != ["0050", "00632R", "0050", "00632R"]
                    or spec.predicted_score_column not in frame.columns
                    or spec.context_column not in frame.columns
                    or not bool(frame[spec.context_column].between(0.0, 1.0).all())
                    or not bool(
                        (
                            frame.loc[
                                frame["context_phase"].eq("selection_crossfit"),
                                "model_information_cutoff",
                            ]
                            < frame.loc[
                                frame["context_phase"].eq("selection_crossfit"), "date"
                            ]
                        ).all()
                    )
                ):
                    context_roundtrip_errors.append(spec.source)

    check(
        "predicted_context_builder_uses_isolated_crossfit_selection_and_single_fixed_forward_behavior",
        [],
        context_orchestration_errors,
    )
    check(
        "predicted_context_artifact_owner_and_stage1_provenance_are_registry_driven",
        [],
        context_owner_errors,
    )
    check(
        "predicted_context_artifact_roundtrip_preserves_pit_and_ticker_identity",
        [],
        context_roundtrip_errors,
    )

    # Rows without a usable persistent context must be filtered before any daily target
    # percentile is derived. This split invariant is shared by all predicted contexts.
    split_dates = pd.to_datetime(
        [f"2018-01-{day:02d}" for day in range(1, 26)]
        + [f"2019-01-{day:02d}" for day in range(1, 26)]
        + [f"2021-01-{day:02d}" for day in range(1, 26)]
    )
    split_target_valid = np.ones(len(split_dates), dtype=bool)
    split_target_valid[[0, 25, 50]] = False
    split_bundle = SimpleNamespace(
        group_table=pd.DataFrame(
            {
                "date": split_dates,
                "label_eval_end_date": split_dates + pd.Timedelta(days=40),
            }
        ),
        target_valid=split_target_valid,
        summary={"training_universe_start_date": "2018-01-01"},
        outer_policy=outer_policy,
    )
    split = build_daily_ranker_split(split_bundle, inner_validation_months=24)
    invalid_ids = {0, 25, 50}
    check_true(
        "predicted_context_training_split_never_readmits_context_missing_rows",
        invalid_ids.isdisjoint(set(split.selection_ids.tolist()))
        and invalid_ids.isdisjoint(set(split.inner_train_ids.tolist()))
        and invalid_ids.isdisjoint(set(split.validation_ids.tolist()))
        and invalid_ids.isdisjoint(set(split.oos_ids.tolist()))
        and bool(split.report.get("target_valid_filter_applied")),
    )


    # Execution recipe is derived from canonical profile/spec but intentionally omits
    # MR identity.  Historical per-profile flags remain frozen evidence; current model
    # workflow authorization is instead derived from the configured Training Profile /
    # Model Compare-Test List SSOT and must not require a second profile authorization list.
    current_required_profiles = set()
    current_strategy_scoped_profiles = set()
    for mode in get_strategy_rolling_test_modes():
        mode_settings = get_strategy_comparison_settings(str(mode["profile_id"]))
        for arm in mode_settings.enabled_arms:
            if not bool(arm.dl_enabled) or not arm.dl_id:
                continue
            required_dl_ids = [str(arm.dl_id)]
            runtime_options = dict(arm.dl_runtime_options or {})
            safety_dl_id = str(runtime_options.get("safety_dl_id") or "").strip()
            if safety_dl_id:
                required_dl_ids.append(safety_dl_id)
            for dl_id in required_dl_ids:
                source = mode_settings.dl_sources[dl_id]
                profile_name = str(source.experiment_profile)
                if profile_name not in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
                    continue
                if bool(source.single_seed_strategy_conversion_authorized):
                    current_strategy_scoped_profiles.add(profile_name)
                else:
                    current_required_profiles.add(profile_name)

    current_dependency_profiles = current_required_profiles | current_strategy_scoped_profiles
    check_true(
        "current_compare_dependencies_use_shared_model_workflow_authorization",
        bool(current_dependency_profiles)
        and all(
            breakout_quality_config.get_breakout_quality_workflow_settings(
                experiment_profile=profile_name
            ).rolling_authorized
            for profile_name in current_dependency_profiles
        ),
    )

    representative_profile = next(
        iter(sorted(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES))
    )
    recipe_keys = set(
        get_continuous_ranker_execution_recipe(representative_profile).as_dict()
    )
    check(
        "execution_recipe_excludes_research_identity_fields",
        set(),
        recipe_keys.intersection({"model_research_id", "experiment_name", "phase"}),
    )

    recipe_mismatches = []
    for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
        profile = get_breakout_quality_experiment_profile(profile_name)
        research_spec = get_continuous_ranker_research_spec(profile_name)
        recipe = get_continuous_ranker_execution_recipe(profile_name)
        observed = (
            recipe.trainer_family,
            recipe.training_objective,
            recipe.continuous_target_id,
            recipe.loss_name,
            recipe.model_architecture,
            recipe.training_label_scope,
            recipe.training_sample_scope,
            recipe.score_semantic_id,
            recipe.pairwise_reduction,
            recipe.historical_pit_authorized,
        )
        expected = (
            research_spec.trainer_family,
            profile.training_objective,
            str(profile.continuous_target_id),
            profile.loss_name,
            profile.model_architecture,
            profile.training_label_scope,
            profile.training_sample_scope,
            research_spec.score_semantic_id,
            research_spec.pairwise_reduction,
            bool(research_spec.selection_pit_authorized),
        )
        if observed != expected:
            recipe_mismatches.append(profile_name)
    check(
        "execution_recipe_is_pure_derivation_without_second_semantics_table",
        [],
        recipe_mismatches,
    )

    authorization_inversions = [
        profile_name
        for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if get_continuous_ranker_execution_recipe(profile_name).current_time_validation_authorized
        and not get_continuous_ranker_execution_recipe(
            profile_name
        ).historical_pit_authorized
    ]
    check(
        "current_time_validation_never_bypasses_historical_pit_authorization",
        [],
        authorization_inversions,
    )

    check(
        "active_and_legacy_model_architecture_sets_are_disjoint_and_exhaustive",
        (set(), set(SUPPORTED_MODEL_ARCHITECTURES)),
        (
            set(ACTIVE_MODEL_ARCHITECTURES).intersection(LEGACY_MODEL_ARCHITECTURES),
            set(ACTIVE_MODEL_ARCHITECTURES).union(LEGACY_MODEL_ARCHITECTURES),
        ),
    )

    check(
        "model_spec_registry_is_exhaustive_without_second_architecture_matrix",
        set(SUPPORTED_MODEL_ARCHITECTURES),
        set(MODEL_SPEC_BUILDERS),
    )
    model_spec_roundtrip_mismatches = []
    for architecture in SUPPORTED_MODEL_ARCHITECTURES:
        spec = get_model_spec(architecture)
        if MODEL_SPEC_BUILDERS[architecture](architecture).as_manifest_payload() != spec.as_manifest_payload():
            model_spec_roundtrip_mismatches.append(f"builder:{architecture}")
            continue
        if model_spec_from_manifest(spec.as_manifest_payload()) != spec:
            model_spec_roundtrip_mismatches.append(f"manifest:{architecture}")
    check(
        "model_spec_registry_builders_roundtrip_all_supported_architectures",
        [],
        model_spec_roundtrip_mismatches,
    )

    descriptor_ids = tuple(ARCHITECTURE_DESCRIPTORS)
    check(
        "architecture_descriptor_registry_is_supported_membership_ssot",
        tuple(SUPPORTED_MODEL_ARCHITECTURES),
        descriptor_ids,
    )
    active_from_descriptors = tuple(
        descriptor.architecture_id
        for descriptor in sorted(
            (value for value in ARCHITECTURE_DESCRIPTORS.values() if value.active),
            key=lambda value: int(value.active_order if value.active_order is not None else 10**9),
        )
    )
    check(
        "active_architecture_order_is_derived_from_descriptors",
        tuple(ACTIVE_MODEL_ARCHITECTURES),
        active_from_descriptors,
    )
    check_true(
        "architecture_descriptor_runtime_builder_keys_are_registered",
        {descriptor.runtime_builder_key for descriptor in ARCHITECTURE_DESCRIPTORS.values()}
        <= set(registered_runtime_builder_keys()),
    )
    check_true(
        "active_descriptors_have_unique_explicit_active_order",
        all(get_architecture_descriptor(name).active_order is not None for name in ACTIVE_MODEL_ARCHITECTURES)
        and len({get_architecture_descriptor(name).active_order for name in ACTIVE_MODEL_ARCHITECTURES})
        == len(ACTIVE_MODEL_ARCHITECTURES),
    )

    full_window_descriptor = get_architecture_descriptor(
        INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1
    )
    full_window_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1)
    ao_shared_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    check_true(
        "full_window_receptive_field_capability_covers_300_bar_input_without_capacity_change",
        full_window_descriptor.has_capability("full_window_receptive_field")
        and int(full_window_spec.receptive_field_bars) >= 300
        and tuple(full_window_spec.inception_kernel_sizes) == tuple(ao_shared_spec.inception_kernel_sizes) == (39, 19, 9)
        and tuple(full_window_spec.inception_module_dilations) == (1, 1, 1, 1, 2, 2)
        and int(full_window_spec.receptive_field_bars) == 305
        and int(full_window_spec.inception_depth) == int(ao_shared_spec.inception_depth)
        and int(full_window_spec.inception_filters) == int(ao_shared_spec.inception_filters)
        and int(full_window_spec.inception_bottleneck_channels) == int(ao_shared_spec.inception_bottleneck_channels)
        and int(full_window_spec.inception_residual_every) == int(ao_shared_spec.inception_residual_every)
        and tuple(full_window_spec.pooling) == tuple(ao_shared_spec.pooling)
        and tuple(ao_shared_spec.inception_module_dilations) == ()
        and int(ao_shared_spec.receptive_field_bars) == 229,
    )

    wide_descriptor = get_architecture_descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1)
    wide_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1)
    ao_model = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    wide_model = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1)
    ao_params = sum(int(parameter.numel()) for parameter in ao_model.parameters() if parameter.requires_grad)
    wide_params = sum(int(parameter.numel()) for parameter in wide_model.parameters() if parameter.requires_grad)
    check_true(
        "wide_capacity_capability_scales_channels_without_rf_or_topology_change",
        wide_descriptor.has_capability("wide_capacity")
        and int(wide_spec.inception_filters) == 64
        and int(wide_spec.inception_bottleneck_channels) == 64
        and int(ao_shared_spec.inception_filters) == 32
        and int(ao_shared_spec.inception_bottleneck_channels) == 32
        and int(wide_spec.receptive_field_bars) == int(ao_shared_spec.receptive_field_bars) == 229
        and tuple(wide_spec.inception_kernel_sizes) == tuple(ao_shared_spec.inception_kernel_sizes) == (39, 19, 9)
        and tuple(wide_spec.inception_module_dilations) == tuple(ao_shared_spec.inception_module_dilations) == ()
        and int(wide_spec.inception_depth) == int(ao_shared_spec.inception_depth) == 6
        and int(wide_spec.inception_residual_every) == int(ao_shared_spec.inception_residual_every) == 3
        and tuple(wide_spec.pooling) == tuple(ao_shared_spec.pooling)
        and ao_params == 473734
        and wide_params == 1885446
        and wide_params > ao_params,
    )

    project_root = Path(__file__).resolve().parents[2]
    generic_architecture_consumers = (
        project_root / "filters" / "breakout_quality" / "models" / "active.py",
        project_root / "filters" / "breakout_quality" / "models" / "runtime_registry.py",
        project_root / "filters" / "breakout_quality" / "models" / "spec_registry.py",
        project_root / "filters" / "breakout_quality" / "models" / "spec_builders.py",
        project_root / "filters" / "breakout_quality" / "models" / "inception_time.py",
        project_root / "filters" / "breakout_quality" / "daily_ranker_data.py",
    )
    identity_leaks = []
    for source_path in generic_architecture_consumers:
        source_text = read_source_text(source_path)
        for architecture in SUPPORTED_MODEL_ARCHITECTURES:
            if architecture in source_text:
                identity_leaks.append(f"{source_path.name}:{architecture}")
    check(
        "generic_architecture_consumers_do_not_hardcode_architecture_ids",
        [],
        identity_leaks,
    )

    synthetic_descriptor = ArchitectureDescriptor(
        architecture_id="synthetic_inception_registry_probe_v1",
        active=True,
        active_order=999,
        spec_builder_key=get_architecture_descriptor(INCEPTION_TIME_V1).spec_builder_key,
        runtime_builder_key=get_architecture_descriptor(INCEPTION_TIME_V1).runtime_builder_key,
        capabilities=get_architecture_descriptor(INCEPTION_TIME_V1).capabilities,
        spec_options=get_architecture_descriptor(INCEPTION_TIME_V1).spec_options,
    )
    with patch.dict(ARCHITECTURE_DESCRIPTORS, {synthetic_descriptor.architecture_id: synthetic_descriptor}):
        synthetic_spec = get_model_spec(synthetic_descriptor.architecture_id)
    check(
        "new_same_family_architecture_resolves_from_one_descriptor_registration",
        synthetic_descriptor.architecture_id,
        synthetic_spec.architecture,
    )

    # The task-specific Safety/MFE topology is a reusable architecture primitive.
    # Exercise its branch split and gradient ownership here rather than pinning a
    # particular MR identity to the model factory.
    task_spec = get_model_spec(INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1)
    task_model = build_active_model(10, 0, architecture=INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1)
    check_true(
        "task_specific_safety_mfe_uses_one_complete_final_residual_group_per_task",
        int(task_spec.inception_depth) == 2 * int(task_spec.inception_residual_every)
        and len(task_model.inception_modules)
        == int(task_spec.inception_depth) - int(task_spec.inception_residual_every)
        and len(task_model.safety_inception_modules) == int(task_spec.inception_residual_every)
        and len(task_model.mfe_inception_modules) == int(task_spec.inception_residual_every),
    )
    torch.manual_seed(20260902)
    task_x = torch.randn((2, 64, 10), dtype=torch.float32)

    def _gradient_total(parameters):
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += float(parameter.grad.detach().abs().sum().item())
        return total

    task_model.zero_grad(set_to_none=True)
    safety_logits, _mfe_logits = task_model.forward_safety_mfe_heads(task_x, None)
    safety_logits[:, 1].sum().backward()
    safety_shared_grad = _gradient_total(task_model.inception_modules.parameters())
    safety_branch_grad = _gradient_total(task_model.safety_inception_modules.parameters())
    safety_to_mfe_grad = _gradient_total(task_model.mfe_inception_modules.parameters())

    task_model.zero_grad(set_to_none=True)
    _safety_logits, mfe_logits = task_model.forward_safety_mfe_heads(task_x, None)
    mfe_logits[:, 1].sum().backward()
    mfe_shared_grad = _gradient_total(task_model.inception_modules.parameters())
    mfe_branch_grad = _gradient_total(task_model.mfe_inception_modules.parameters())
    mfe_to_safety_grad = _gradient_total(task_model.safety_inception_modules.parameters())
    check_true(
        "task_specific_safety_mfe_gradient_ownership_is_shared_stem_plus_own_final_group",
        safety_shared_grad > 0.0
        and safety_branch_grad > 0.0
        and safety_to_mfe_grad == 0.0
        and mfe_shared_grad > 0.0
        and mfe_branch_grad > 0.0
        and mfe_to_safety_grad == 0.0,
    )

    # Safety-specific temporal attention pooling is another reusable InceptionTime
    # primitive.  The AO-shared trunk/MFE path must remain exact at initialization;
    # only the Safety pooling scorer is new, and MFE loss must not update it.
    ao_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    safety_attn_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1)
    ao_manifest = ao_spec.as_manifest_payload()
    attn_manifest = safety_attn_spec.as_manifest_payload()
    architecture_only_fields = {"architecture", "family", "pooling"}
    check_true(
        "safety_attention_pool_keeps_ao_trunk_and_head_spec_except_pooling_identity",
        all(
            ao_manifest[key] == attn_manifest[key]
            for key in ao_manifest
            if key not in architecture_only_fields
        )
        and tuple(safety_attn_spec.pooling)
        == (
            "safety_scalar_attention_pool",
            "mfe_global_average",
            "raw_safety_head",
            "raw_mfe_head",
        ),
    )
    torch.manual_seed(20260902)
    ao_model = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    torch.manual_seed(20260902)
    safety_attn_model = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1)
    ao_state = ao_model.state_dict()
    attn_state = safety_attn_model.state_dict()
    shared_keys = sorted(set(ao_state).intersection(attn_state))
    extra_keys = sorted(set(attn_state).difference(ao_state))
    check_true(
        "safety_attention_pool_same_seed_adds_only_scalar_scorer_parameters",
        not (set(ao_state) - set(attn_state))
        and extra_keys
        == ["safety_attention_scorer.bias", "safety_attention_scorer.weight"]
        and all(torch.equal(ao_state[key], attn_state[key]) for key in shared_keys),
    )
    scorer = safety_attn_model.safety_attention_scorer
    check_true(
        "safety_attention_pool_is_single_scalar_1x1_scorer_without_attention_hyperparameters",
        isinstance(scorer, torch.nn.Conv1d)
        and int(scorer.in_channels) == int(safety_attn_spec.inception_filters) * 4
        and int(scorer.out_channels) == 1
        and tuple(scorer.kernel_size) == (1,)
        and tuple(scorer.stride) == (1,)
        and tuple(scorer.padding) == (0,),
    )
    torch.manual_seed(20260903)
    attention_x = torch.randn((3, 64, 10), dtype=torch.float32)
    ao_model.eval()
    safety_attn_model.eval()
    with torch.no_grad():
        ao_safety_logits, ao_mfe_logits = ao_model.forward_safety_mfe_heads(attention_x, None)
        attn_safety_logits, attn_mfe_logits = safety_attn_model.forward_safety_mfe_heads(attention_x, None)
        safety_weights = safety_attn_model.safety_attention_weights(attention_x)
    check_true(
        "safety_attention_pool_preserves_ao_mfe_path_and_normalizes_temporal_weights",
        torch.equal(ao_mfe_logits, attn_mfe_logits)
        and tuple(safety_weights.shape) == (3, 64)
        and bool(torch.allclose(safety_weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0.0))
        and bool(torch.isfinite(safety_weights).all())
        and not torch.equal(ao_safety_logits, attn_safety_logits),
    )
    safety_attn_model.zero_grad(set_to_none=True)
    safety_logits, _mfe_logits = safety_attn_model.forward_safety_mfe_heads(attention_x, None)
    safety_logits[:, 1].sum().backward()
    scorer_from_safety_grad = _gradient_total(safety_attn_model.safety_attention_scorer.parameters())
    safety_attn_model.zero_grad(set_to_none=True)
    _safety_logits, mfe_logits = safety_attn_model.forward_safety_mfe_heads(attention_x, None)
    mfe_logits[:, 1].sum().backward()
    scorer_from_mfe_grad = _gradient_total(safety_attn_model.safety_attention_scorer.parameters())
    check_true(
        "safety_attention_pool_gradient_is_safety_only_while_mfe_keeps_shared_gap",
        scorer_from_safety_grad > 0.0 and scorer_from_mfe_grad == 0.0,
    )

    # Composition cell: task-specific high-level representation + the same scalar
    # Safety temporal-attention primitive.  Relative to the task-specific GAP model,
    # only the Safety pooling scorer may be new; MFE branch topology/logits must stay exact.
    task_attn_spec = get_model_spec(INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1)
    task_manifest = task_spec.as_manifest_payload()
    task_attn_manifest = task_attn_spec.as_manifest_payload()
    check_true(
        "task_specific_safety_attention_composes_existing_primitives_without_new_hyperparameters",
        all(
            task_manifest[key] == task_attn_manifest[key]
            for key in task_manifest
            if key not in architecture_only_fields
        )
        and tuple(task_attn_spec.pooling)
        == (
            "task_specific_final_residual_group",
            "safety_scalar_attention_pool",
            "mfe_global_average",
            "raw_safety_head",
            "raw_mfe_head",
        ),
    )
    torch.manual_seed(20260902)
    task_gap_model = build_active_model(10, 0, architecture=INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1)
    torch.manual_seed(20260902)
    task_attn_model = build_active_model(10, 0, architecture=INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1)
    task_gap_state = task_gap_model.state_dict()
    task_attn_state = task_attn_model.state_dict()
    task_shared_keys = sorted(set(task_gap_state).intersection(task_attn_state))
    task_extra_keys = sorted(set(task_attn_state).difference(task_gap_state))
    check_true(
        "task_specific_safety_attention_same_seed_adds_only_scalar_scorer",
        not (set(task_gap_state) - set(task_attn_state))
        and task_extra_keys
        == ["safety_attention_scorer.bias", "safety_attention_scorer.weight"]
        and all(torch.equal(task_gap_state[key], task_attn_state[key]) for key in task_shared_keys),
    )
    task_gap_model.eval()
    task_attn_model.eval()
    with torch.no_grad():
        gap_safety_logits, gap_mfe_logits = task_gap_model.forward_safety_mfe_heads(attention_x, None)
        task_attn_safety_logits, task_attn_mfe_logits = task_attn_model.forward_safety_mfe_heads(attention_x, None)
        task_safety_weights = task_attn_model.safety_attention_weights(attention_x)
        safety_map, _mfe_map = task_attn_model.encode_task_specific_feature_maps(attention_x)
        expected_task_weights = torch.softmax(
            task_attn_model.safety_attention_scorer(safety_map).squeeze(1).float(), dim=1
        ).to(safety_map.dtype)
    check_true(
        "task_specific_safety_attention_reads_safety_map_and_preserves_mfe_path",
        torch.equal(gap_mfe_logits, task_attn_mfe_logits)
        and bool(torch.allclose(task_safety_weights, expected_task_weights, atol=1e-6, rtol=0.0))
        and bool(torch.allclose(task_safety_weights.sum(dim=1), torch.ones(3), atol=1e-6, rtol=0.0))
        and not torch.equal(gap_safety_logits, task_attn_safety_logits),
    )
    task_attn_model.zero_grad(set_to_none=True)
    task_attn_safety_logits, _task_attn_mfe_logits = task_attn_model.forward_safety_mfe_heads(attention_x, None)
    task_attn_safety_logits[:, 1].sum().backward()
    task_safety_scorer_grad = _gradient_total(task_attn_model.safety_attention_scorer.parameters())
    task_safety_branch_grad = _gradient_total(task_attn_model.safety_inception_modules.parameters())
    task_safety_to_mfe_grad = _gradient_total(task_attn_model.mfe_inception_modules.parameters())
    task_attn_model.zero_grad(set_to_none=True)
    _task_attn_safety_logits, task_attn_mfe_logits = task_attn_model.forward_safety_mfe_heads(attention_x, None)
    task_attn_mfe_logits[:, 1].sum().backward()
    task_mfe_to_scorer_grad = _gradient_total(task_attn_model.safety_attention_scorer.parameters())
    task_mfe_to_safety_branch_grad = _gradient_total(task_attn_model.safety_inception_modules.parameters())
    task_mfe_branch_grad = _gradient_total(task_attn_model.mfe_inception_modules.parameters())
    check_true(
        "task_specific_safety_attention_gradient_ownership_remains_task_isolated",
        task_safety_scorer_grad > 0.0
        and task_safety_branch_grad > 0.0
        and task_safety_to_mfe_grad == 0.0
        and task_mfe_to_scorer_grad == 0.0
        and task_mfe_to_safety_branch_grad == 0.0
        and task_mfe_branch_grad > 0.0,
    )

    # Safety temporal self-attention is a distinct interaction primitive, not another
    # pooling variant.  AO feature extraction/MFE GAP remain exact; only Safety receives
    # single-head time-to-time Q/K/V interaction followed by the unchanged GAP.
    self_attn_spec = get_model_spec(INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1)
    self_attn_manifest = self_attn_spec.as_manifest_payload()
    check_true(
        "safety_temporal_self_attention_keeps_ao_spec_except_interaction_identity",
        all(
            ao_manifest[key] == self_attn_manifest[key]
            for key in ao_manifest
            if key not in architecture_only_fields
        )
        and tuple(self_attn_spec.pooling)
        == (
            "safety_single_head_temporal_self_attention_residual",
            "safety_global_average",
            "mfe_global_average",
            "raw_safety_head",
            "raw_mfe_head",
        ),
    )
    torch.manual_seed(20260902)
    ao_self_control = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    torch.manual_seed(20260902)
    self_attn_model = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1)
    ao_self_state = ao_self_control.state_dict()
    self_attn_state = self_attn_model.state_dict()
    self_attn_shared_keys = sorted(set(ao_self_state).intersection(self_attn_state))
    self_attn_extra_keys = sorted(set(self_attn_state).difference(ao_self_state))
    check_true(
        "safety_temporal_self_attention_same_seed_adds_only_qkv_projections",
        not (set(ao_self_state) - set(self_attn_state))
        and self_attn_extra_keys
        == [
            "safety_temporal_key.weight",
            "safety_temporal_query.weight",
            "safety_temporal_value.weight",
        ]
        and all(torch.equal(ao_self_state[key], self_attn_state[key]) for key in self_attn_shared_keys),
    )
    q_proj = self_attn_model.safety_temporal_query
    k_proj = self_attn_model.safety_temporal_key
    v_proj = self_attn_model.safety_temporal_value
    expected_channels = int(self_attn_spec.inception_filters) * 4
    check_true(
        "safety_temporal_self_attention_is_parameter_minimal_single_head_full_width_qkv",
        all(
            isinstance(layer, torch.nn.Conv1d)
            and int(layer.in_channels) == expected_channels
            and int(layer.out_channels) == expected_channels
            and tuple(layer.kernel_size) == (1,)
            and layer.bias is None
            for layer in (q_proj, k_proj, v_proj)
        )
        and not hasattr(self_attn_model, "safety_temporal_ffn")
        and not hasattr(self_attn_model, "safety_positional_embedding"),
    )
    ao_self_control.eval()
    self_attn_model.eval()
    with torch.no_grad():
        ao_self_safety, ao_self_mfe = ao_self_control.forward_safety_mfe_heads(attention_x, None)
        self_attn_safety, self_attn_mfe = self_attn_model.forward_safety_mfe_heads(attention_x, None)
    check_true(
        "safety_temporal_self_attention_preserves_ao_mfe_path_and_changes_only_safety_readout",
        torch.equal(ao_self_mfe, self_attn_mfe)
        and not torch.equal(ao_self_safety, self_attn_safety),
    )
    self_attn_model.zero_grad(set_to_none=True)
    self_attn_safety, _self_attn_mfe = self_attn_model.forward_safety_mfe_heads(attention_x, None)
    self_attn_safety[:, 1].sum().backward()
    safety_to_qkv_grad = sum(
        _gradient_total(layer.parameters()) for layer in (q_proj, k_proj, v_proj)
    )
    self_attn_model.zero_grad(set_to_none=True)
    _self_attn_safety, self_attn_mfe = self_attn_model.forward_safety_mfe_heads(attention_x, None)
    self_attn_mfe[:, 1].sum().backward()
    mfe_to_qkv_grad = sum(
        _gradient_total(layer.parameters()) for layer in (q_proj, k_proj, v_proj)
    )
    check_true(
        "safety_temporal_self_attention_gradient_is_safety_only",
        safety_to_qkv_grad > 0.0 and mfe_to_qkv_grad == 0.0,
    )

    # Dual-encoder architecture: Safety gets the frozen historical Patch recipe while
    # Conditional-MFE keeps the AO-form InceptionTime path.  The two optimizers are still
    # one model/one loss composition, but gradient ownership is intentionally encoder-local.
    hybrid_spec = get_model_spec(PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1)
    hybrid_manifest = hybrid_spec.as_manifest_payload()
    check_true(
        "safety_patch_mfe_inception_spec_combines_frozen_patch_recipe_and_ao_inception_recipe",
        int(hybrid_spec.patch_transformer_patch_size) == 10
        and int(hybrid_spec.patch_transformer_patch_stride) == 10
        and int(hybrid_spec.patch_transformer_embedding_dim) == 128
        and int(hybrid_spec.patch_transformer_depth) == 3
        and int(hybrid_spec.patch_transformer_heads) == 4
        and int(hybrid_spec.patch_transformer_mlp_dim) == 256
        and str(hybrid_spec.patch_transformer_positional_encoding) == "sinusoidal"
        and abs(float(hybrid_spec.dropout) - 0.10) < 1e-12
        and int(hybrid_spec.inception_depth) == int(ao_spec.inception_depth)
        and tuple(hybrid_spec.inception_kernel_sizes) == tuple(ao_spec.inception_kernel_sizes)
        and tuple(hybrid_spec.pooling) == (
            "safety_patch_token_global_average",
            "mfe_inception_global_average",
            "raw_safety_head",
            "raw_mfe_head",
        ),
    )
    topology = hybrid_spec.hs_conditional_mfe_topology_contract() or {}
    check_true(
        "safety_patch_mfe_inception_topology_declares_independent_encoder_gradient_ownership",
        topology.get("architecture")
        == "independent_patch_transformer_safety_encoder_plus_inceptiontime_conditional_mfe_encoder"
        and topology.get("conditional_mfe_head_inputs")
        == "inceptiontime_latent_only_no_predicted_safety_context"
        and topology.get("gradient_ownership")
        == "safety_loss_updates_patch_encoder_only_conditional_mfe_loss_updates_inceptiontime_encoder_only",
    )
    torch.manual_seed(20260904)
    hybrid_x = torch.randn((3, 300, 10), dtype=torch.float32)
    torch.manual_seed(20260902)
    ao_hybrid_control = build_active_model(10, 0, architecture=INCEPTION_TIME_SHARED_SAFETY_MFE_V1)
    torch.manual_seed(20260902)
    hybrid_model = build_active_model(10, 0, architecture=PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1)
    ao_hybrid_state = ao_hybrid_control.state_dict()
    hybrid_state = hybrid_model.state_dict()
    check_true(
        "safety_patch_mfe_inception_same_seed_preserves_complete_ao_mfe_model_initialization",
        all(
            "mfe_model." + key in hybrid_state
            and torch.equal(value, hybrid_state["mfe_model." + key])
            for key, value in ao_hybrid_state.items()
        ),
    )
    ao_hybrid_control.eval()
    hybrid_model.eval()
    with torch.no_grad():
        _ao_safety, ao_hybrid_mfe = ao_hybrid_control.forward_safety_mfe_heads(hybrid_x, None)
        hybrid_safety, hybrid_mfe = hybrid_model.forward_safety_mfe_heads(hybrid_x, None)
        hybrid_both = hybrid_model.forward_output_head(hybrid_x, None, "conditional_both")
    check_true(
        "safety_patch_mfe_inception_preserves_ao_mfe_logits_and_exposes_duo_head_output",
        torch.equal(ao_hybrid_mfe, hybrid_mfe)
        and tuple(hybrid_safety.shape) == (3, 2)
        and tuple(hybrid_both.shape) == (3, 4)
        and bool(torch.isfinite(hybrid_safety).all())
        and bool(torch.isfinite(hybrid_mfe).all()),
    )
    hybrid_model.train()
    hybrid_model.zero_grad(set_to_none=True)
    hybrid_safety, _hybrid_mfe = hybrid_model.forward_safety_mfe_heads(hybrid_x, None)
    hybrid_safety[:, 1].sum().backward()
    safety_patch_grad = _gradient_total(hybrid_model.safety_encoder.parameters()) + _gradient_total(
        hybrid_model.raw_safety_classifier.parameters()
    )
    safety_to_mfe_encoder_grad = _gradient_total(hybrid_model.mfe_model.parameters())
    hybrid_model.zero_grad(set_to_none=True)
    _hybrid_safety, hybrid_mfe = hybrid_model.forward_safety_mfe_heads(hybrid_x, None)
    hybrid_mfe[:, 1].sum().backward()
    mfe_to_patch_grad = _gradient_total(hybrid_model.safety_encoder.parameters()) + _gradient_total(
        hybrid_model.raw_safety_classifier.parameters()
    )
    mfe_encoder_grad = _gradient_total(hybrid_model.mfe_model.parameters())
    check_true(
        "safety_patch_mfe_inception_gradient_ownership_is_encoder_isolated",
        safety_patch_grad > 0.0
        and safety_to_mfe_encoder_grad == 0.0
        and mfe_to_patch_grad == 0.0
        and mfe_encoder_grad > 0.0,
    )

    daily_trainer_source = read_source_text(
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    )
    continuous_trainer_source = read_source_text(
        project_root / "services" / "breakout_quality" / "train_continuous_ranker.py"
    )
    check_true(
        "training_report_consumes_canonical_training_semantics_without_contract_whitelist",
        "**canonical_training_semantics" in daily_trainer_source
        and "**canonical_training_semantics" in continuous_trainer_source
        and '"shared_safety_weighted_mfe_duo_head_contract": ranker_api.training_semantics' not in daily_trainer_source,
    )

    summary["components"] = (
        "dual_component_regression",
        "equal_rank_target",
        "risk_normalized_context",
        "pareto_pairwise",
        "predicted_context_artifact",
    )
    summary["training_performed"] = False
    return results, summary



def validate_breakout_quality_low_adverse_target_component_contract_case(_base_params):
    """Protect the reusable full-horizon negative-adverse target transform."""
    case_id = "BREAKOUT_QUALITY_LOW_ADVERSE_TARGET_COMPONENT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from filters.breakout_quality.continuous_target import (
        DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
        build_daily_full_horizon_low_adverse_contract,
        daily_full_horizon_low_adverse_target_from_cached_path,
    )
    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    horizon = int(spec.horizon_bars)
    highs = np.linspace(101.0, 120.0, horizon, dtype=np.float64)
    lows = np.full(horizon, 98.0, dtype=np.float64)
    lows[5] = 94.0
    source = daily_full_horizon_opportunity_target_from_cached_path(
        highs, lows, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    adverse = daily_full_horizon_low_adverse_target_from_cached_path(
        highs, lows, anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "low_adverse_target_reuses_peak_diagnostics_and_is_negative_adverse_r",
        (
            source.opportunity_bar, source.first_risk_breach_bar,
            round(float(source.favorable_return), 6), round(float(source.adverse_return_to_peak), 6),
            round(-float(source.adverse_return_to_peak) / float(spec.risk_budget_return), 6),
        ),
        (
            adverse.opportunity_bar, adverse.first_risk_breach_bar,
            round(float(adverse.favorable_return), 6), round(float(adverse.adverse_return_to_peak), 6),
            round(float(adverse.target_raw_r), 6),
        ),
    )
    dates = pd.date_range("2020-01-01", periods=horizon + 3, freq="D")
    frame = pd.DataFrame({
        "Open": np.full(len(dates), 100.0), "High": np.concatenate(([100.0], highs, [102.0, 102.0])),
        "Low": np.concatenate(([100.0], lows, [99.0, 99.0])), "Close": np.full(len(dates), 100.0),
        "Volume": np.full(len(dates), 1000.0),
    }, index=dates)
    batch = compute_daily_opportunity_target_batch(
        frame, np.asarray([0], dtype=np.int64), spec=spec, target_id=DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID,
    )
    check("low_adverse_target_scalar_vector_parity", round(float(adverse.target_raw_r), 6), round(float(batch.target_raw_r[0]), 6))
    contract = build_daily_full_horizon_low_adverse_contract(DEFAULT_LABEL_POLICY)
    check(
        "low_adverse_target_contract_is_strategy_agnostic_without_mfe_reward",
        (DAILY_FULL_HORIZON_LOW_ADVERSE_TARGET_ID, False, True, False, False),
        (
            contract["target_id"], bool(contract["mfe_reward_included"]), bool(contract["higher_is_better"]),
            bool(contract["requires_strategy_candidate_membership"]), "favorable_return /" in str(contract["formula"]),
        ),
    )
    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_conditional_mfe_safety_single_model_contract_case(_base_params):
    """Protect MR-13P single-model conditional target/head semantics."""

    case_id = "BREAKOUT_QUALITY_CONDITIONAL_MFE_SAFETY_SINGLE_MODEL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_safety import (
        build_conditional_mfe_safety_targets,
    )
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from filters.breakout_quality.models.runtime import require_torch

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    check(
        "mr13p_identity_objective_and_architecture_are_explicit",
        (
            "MR-13P",
            TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
            "inception_time_conditional_mfe_safety_v1",
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
        ),
    )
    semantics = training_semantics(profile)
    conditional_contract = dict(semantics.get("conditional_mfe_safety_contract") or {})
    check(
        "mr13p_uses_full_list_dual_head_equal_weight_stop_gradient_contract",
        (
            CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            "fixed_equal_mean_no_lambda_sweep",
            "stop_gradient_primary_mfe_probability",
            False,
            True,
        ),
        (
            recipe.pairwise_reduction,
            conditional_contract.get("head_weighting"),
            conditional_contract.get("conditional_context"),
            conditional_contract.get("primary_head_gradient_from_conditional_loss"),
            conditional_contract.get("shared_encoder_gradient_from_both_heads"),
        ),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 5 + ["2026-01-05"] * 5),
            "target_favorable_r": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "target_adverse_r": [0.1, 0.4, 0.3, 0.8, 0.5, 0.2, 0.2, 0.7, 0.6, 0.9],
        }
    )
    valid = np.ones(len(frame), dtype=bool)
    targets = build_conditional_mfe_safety_targets(frame, valid)
    check("mr13p_training_target_has_two_same_scale_heads", (10, 2), targets.training_target.shape)
    check_true(
        "mr13p_conditional_target_is_finite_unit_interval",
        bool(
            np.isfinite(targets.training_target).all()
            and (targets.training_target >= 0.0).all()
            and (targets.training_target <= 1.0).all()
        ),
    )
    orthogonal = True
    for _date, day in frame.groupby("date", sort=True):
        idx = day.index.to_numpy(dtype=np.int64)
        u = targets.primary_mfe_percentile[idx].astype(np.float64)
        residual = targets.conditional_safety_residual[idx].astype(np.float64)
        orthogonal &= abs(float(np.mean(residual))) < 1e-6
        orthogonal &= abs(float(np.dot(u - np.mean(u), residual))) < 1e-6
    check_true(
        "mr13p_conditional_residual_removes_same_date_linear_mfe_relation",
        orthogonal,
    )

    torch, _nn = require_torch()
    torch.manual_seed(7)
    model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_conditional_mfe_safety_v1",
    )
    x = torch.randn(4, 300, 10)
    context = torch.empty(4, 0)
    primary_logits, safety_logits = model.forward_conditional_heads(x, context)
    check(
        "mr13p_one_model_exposes_two_two_logit_heads",
        ((4, 2), (4, 2)),
        (tuple(primary_logits.shape), tuple(safety_logits.shape)),
    )
    model.zero_grad(set_to_none=True)
    safety_logits.sum().backward()
    primary_grad = model.classifier.weight.grad
    shared_grad = next(model.inception_modules[0].parameters()).grad
    check_true(
        "mr13p_safety_loss_cannot_backprop_into_primary_head_but_updates_shared_encoder",
        primary_grad is None and shared_grad is not None,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_reverse_conditional_mfe_ab_contract_case(_base_params):
    """Protect MR-13Q/R common reverse-conditional target and only architecture difference."""

    case_id = "BREAKOUT_QUALITY_REVERSE_CONDITIONAL_MFE_AB"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_opportunity import (
        build_conditional_mfe_opportunity_targets,
    )
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from filters.breakout_quality.models.runtime import require_torch

    single = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    duo = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    single_spec = get_continuous_ranker_research_spec(single.name)
    duo_spec = get_continuous_ranker_research_spec(duo.name)
    check(
        "reverse_conditional_ab_identity_objective_and_architecture_are_controlled",
        (
            "MR-13Q", TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_PAIRWISE_RANKING, "inception_time_v1",
            "MR-13R", TRAINING_OBJECTIVE_DAILY_SAFETY_CONDITIONAL_MFE_PAIRWISE_RANKING,
            "inception_time_safety_conditional_mfe_v1",
        ),
        (
            single_spec.model_research_id, single.training_objective, single.model_architecture,
            duo_spec.model_research_id, duo.training_objective, duo.model_architecture,
        ),
    )
    single_sem = dict(training_semantics(single).get("conditional_mfe_single_head_contract") or {})
    duo_sem = dict(training_semantics(duo).get("safety_conditional_mfe_duo_head_contract") or {})
    check(
        "reverse_conditional_ab_share_same_final_target_and_strategy_score_semantics",
        (
            "same_date_percentile_of_pure_mfe_residual_after_same_date_OLS_on_true_low_adverse_safety_percentile",
            "conditional_mfe_pass_probability",
            "same_date_percentile_of_pure_mfe_residual_after_same_date_OLS_on_true_low_adverse_safety_percentile",
            "conditional_mfe_pass_probability_only",
        ),
        (
            single_sem.get("target"), single_sem.get("runtime_score"),
            duo_sem.get("conditional_mfe_target"), duo_sem.get("runtime_score"),
        ),
    )
    check(
        "duo_head_uses_raw_safety_only_as_stop_gradient_condition",
        ("same_date_low_adverse_safety_percentile", "stop_gradient_raw_safety_probability", False),
        (
            duo_sem.get("safety_target"), duo_sem.get("conditional_context"),
            duo_sem.get("safety_head_gradient_from_conditional_loss"),
        ),
    )
    check_true(
        "both_reverse_conditional_profiles_are_research_authorized_for_pit_conversion",
        bool(single_spec.selection_pit_authorized and duo_spec.selection_pit_authorized),
    )

    project_root = Path(__file__).resolve().parents[2]
    app_source = (
        project_root / "services" / "research" / "breakout_quality_application.py"
    ).read_text(encoding="utf-8")
    check_true(
        "reverse_conditional_profiles_share_config_driven_standard_model_comparison_route_without_dedicated_menu",
        'model_list = get_breakout_quality_model_test_settings().model_profiles' in app_source
        and 'for model_id, profile_name in model_list:' in app_source
        and 'render_menu_item(3, "Forward OOS 模型比較")' in app_source
        and 'render_menu_item(4, "Rolling OOS 模型比較")' in app_source
        and 'if choice == "3":' in app_source
        and '_run_configured_model_comparison(program_name, rolling=False)' in app_source
        and '_run_configured_model_comparison(program_name, rolling=True)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=False)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=True)' in app_source
        and '_run_shared_strategy_robustness' not in app_source
        and 'Conditional-MFE Single／Duo Forward Model Gate' not in app_source,
    )
    daily_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    oos_frame_init = 'oos_frame = bundle.group_table.iloc[forward_score_ids][["ticker", "date", "group_index"]].copy()'
    check_true(
        "reverse_conditional_forward_score_columns_are_written_only_after_oos_frame_initialization",
        oos_frame_init in daily_source
        and daily_source.index(oos_frame_init) < daily_source.index('oos_frame["conditional_mfe_score"]')
        and daily_source.index(oos_frame_init) < daily_source.index('oos_frame["raw_safety_score"]'),
    )
    check(
        "both_reverse_conditional_profiles_use_same_full_list_pairwise_reduction",
        get_continuous_ranker_execution_recipe(single.name).pairwise_reduction,
        get_continuous_ranker_execution_recipe(duo.name).pairwise_reduction,
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 6 + ["2026-01-05"] * 6),
            "target_favorable_r": [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "target_adverse_r": [0.2, 0.8, 0.4, 1.0, 0.5, 0.9, 0.9, 0.2, 0.7, 0.3, 0.8, 0.4],
        }
    )
    targets = build_conditional_mfe_opportunity_targets(frame, np.ones(len(frame), dtype=bool))
    check("single_head_training_target_is_one_final_j", (12,), targets.single_head_training_target.shape)
    check("duo_head_training_target_is_raw_safety_plus_same_final_j", (12, 2), targets.duo_head_training_target.shape)
    check_true(
        "reverse_conditional_targets_are_finite_unit_interval",
        bool(np.isfinite(targets.duo_head_training_target).all()
             and (targets.duo_head_training_target >= 0).all()
             and (targets.duo_head_training_target <= 1).all()),
    )
    check_true(
        "single_and_duo_final_conditional_mfe_targets_are_identical",
        bool(np.allclose(targets.single_head_training_target, targets.duo_head_training_target[:, 1], atol=0, rtol=0)),
    )
    orthogonal = True
    for _date, day in frame.groupby("date", sort=True):
        idx = day.index.to_numpy(dtype=np.int64)
        s = targets.low_adverse_safety_percentile[idx].astype(np.float64)
        residual = targets.conditional_mfe_residual[idx].astype(np.float64)
        orthogonal &= abs(float(np.mean(residual))) < 1e-6
        orthogonal &= abs(float(np.dot(s - np.mean(s), residual))) < 1e-6
    check_true("reverse_conditional_residual_removes_same_date_linear_safety_relation", orthogonal)

    torch, _nn = require_torch()
    torch.manual_seed(11)
    model = build_active_model(feature_count=10, context_count=0, architecture="inception_time_safety_conditional_mfe_v1")
    x = torch.randn(4, 300, 10)
    context = torch.empty(4, 0)
    safety_logits, conditional_logits = model.forward_safety_conditional_mfe_heads(x, context)
    check(
        "duo_head_model_exposes_raw_safety_and_final_conditional_mfe_heads",
        ((4, 2), (4, 2), (4, 2)),
        (tuple(safety_logits.shape), tuple(conditional_logits.shape), tuple(model(x, context).shape)),
    )
    model.zero_grad(set_to_none=True)
    conditional_logits.sum().backward()
    safety_grad = model.raw_safety_classifier.weight.grad
    shared_grad = next(model.inception_modules[0].parameters()).grad
    check_true(
        "conditional_mfe_loss_cannot_rewrite_raw_safety_head_but_updates_shared_encoder",
        safety_grad is None and shared_grad is not None,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_safety_raw_mfe_duo_contract_case(_base_params):
    """Protect MR-13S as the single-target controlled contrast to MR-13R."""

    case_id = "BREAKOUT_QUALITY_SAFETY_RAW_MFE_DUO"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_opportunity import (
        build_conditional_mfe_opportunity_targets,
    )
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.ranker_training import safety_raw_mfe_metrics
    from services.breakout_quality.train_daily_ranker import (
        build_safety_raw_mfe_truth_geometry_control_from_score_frame,
    )

    old_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_CONDITIONAL_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13s_identity_is_preserved_as_model_only_controlled_contrast",
        (
            DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "MR-13S",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_PAIRWISE_RANKING,
            "inception_time_safety_conditional_mfe_v1",
            False,
            False,
        ),
        (
            profile.name,
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check(
        "mr13s_changes_target_not_architecture_or_pairwise_reduction",
        (
            old_profile.model_architecture,
            get_continuous_ranker_execution_recipe(old_profile.name).pairwise_reduction,
        ),
        (
            profile.model_architecture,
            get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction,
        ),
    )
    sem = dict(training_semantics(profile).get("safety_raw_mfe_duo_head_contract") or {})
    check(
        "mr13s_contract_is_raw_safety_plus_absolute_mfe_with_stop_gradient_context",
        (
            "same_date_low_adverse_safety_percentile",
            "same_date_pure_mfe_percentile",
            "stop_gradient_raw_safety_probability",
            False,
            "fixed_equal_mean_no_lambda_sweep",
            "raw_mfe_mean_daily_spearman",
            "model_gate_only_no_pit_no_strategy_conversion",
        ),
        (
            sem.get("safety_target"),
            sem.get("raw_mfe_target"),
            sem.get("conditional_context"),
            sem.get("safety_head_gradient_from_mfe_loss"),
            sem.get("head_weighting"),
            sem.get("epoch_selection"),
            sem.get("runtime_status"),
        ),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 10 + ["2026-01-05"] * 10),
            "target_favorable_r": list(range(1, 11)) * 2,
            "target_adverse_r": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 1.0, 0.5, 0.6, 0.1] * 2,
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
        }
    )
    targets = build_conditional_mfe_opportunity_targets(
        frame, np.ones(len(frame), dtype=bool)
    )
    new_target = targets.safety_raw_mfe_training_target
    old_target = targets.duo_head_training_target
    check("mr13s_training_target_shape", (20, 2), new_target.shape)
    check_true(
        "mr13s_safety_target_is_byte_identical_to_mr13r_safety_target",
        bool(np.array_equal(new_target[:, 0], old_target[:, 0])),
    )
    check_true(
        "mr13s_final_target_is_absolute_u_not_residual_j",
        bool(
            np.array_equal(new_target[:, 1], targets.primary_mfe_percentile)
            and not np.array_equal(new_target[:, 1], old_target[:, 1])
        ),
    )

    # Controlled perfect marginal scores must create supported upper-right geometry.
    scores = {
        "raw_safety": targets.low_adverse_safety_percentile.astype(np.float32),
        "raw_mfe": targets.primary_mfe_percentile.astype(np.float32),
    }
    metrics = safety_raw_mfe_metrics(
        np.arange(len(frame), dtype=np.int64), frame, targets, scores
    )
    gate = dict(metrics.get("model_gate") or {})
    upper = dict(gate.get("upper_right_s5_m5") or {})
    geometry = list(gate.get("predicted_joint_geometry") or [])
    cohorts = list(gate.get("safety_cohorts") or [])
    check_true(
        "mr13s_model_gate_has_full_5x5_geometry_and_supported_upper_right",
        len(geometry) == 5
        and all(len(row) == 5 for row in geometry)
        and int(upper.get("n", 0) or 0) > 0
        and float(upper.get("actual_hmhs_pct") or 0.0) == 100.0,
    )
    check_true(
        "mr13s_model_gate_reports_all_safety_cohorts_and_joint_rank",
        len(cohorts) == 5
        and all("raw_mfe_to_actual_mfe_mean_daily_spearman" in row for row in cohorts)
        and gate.get("joint_product_to_actual_hmhs_mean_daily_spearman") is not None,
    )
    truth = dict(gate.get("actual_truth_geometry") or {})
    check_true(
        "mr13s_model_gate_exposes_actual_truth_geometry_and_predicted_head_relation",
        len(list(truth.get("actual_joint_geometry") or [])) == 5
        and gate.get("predicted_safety_to_raw_mfe_mean_daily_spearman") is not None
        and truth.get("safety_to_mfe_mean_daily_spearman") is not None,
    )

    score_frame = pd.DataFrame({
        "ticker": [f"T{i:02d}" for i in range(len(frame))],
        "date": frame["date"],
        "target_low_adverse_safety_percentile": targets.low_adverse_safety_percentile,
        "target_pure_mfe_percentile": targets.primary_mfe_percentile,
        "raw_safety_score": scores["raw_safety"],
        "raw_mfe_score": scores["raw_mfe"],
    })
    candidate_positions = np.asarray([0, 1, 8, 9, 10, 11, 18, 19], dtype=np.int64)
    candidate_keys = {
        (str(score_frame.iloc[pos]["ticker"]), pd.Timestamp(score_frame.iloc[pos]["date"]).normalize())
        for pos in candidate_positions
    }
    truth_control = build_safety_raw_mfe_truth_geometry_control_from_score_frame(
        score_frame,
        breakout_candidate_keys=candidate_keys,
    )
    breakout_truth = dict((truth_control.get("breakout_candidate_oos") or {}).get("actual") or {})
    expected_counts = np.zeros((5, 5), dtype=np.int64)
    for pos in candidate_positions:
        safety_q = min(4, int(float(targets.low_adverse_safety_percentile[pos]) * 5.0))
        mfe_q = min(4, int(float(targets.primary_mfe_percentile[pos]) * 5.0))
        expected_counts[safety_q, mfe_q] += 1
    actual_counts = np.asarray([
        [int(dict(cell or {}).get("n", 0) or 0) for cell in row]
        for row in list(breakout_truth.get("actual_joint_geometry") or [])
    ], dtype=np.int64)
    check_true(
        "mr13s_truth_geometry_breakout_filters_daily_percentiles_without_subset_rerank",
        int((truth_control.get("breakout_candidate_oos") or {}).get("population_n", 0) or 0) == len(candidate_positions)
        and truth_control.get("breakout_percentile_policy") == "filter_daily_universal_percentiles_without_subset_rerank"
        and np.array_equal(actual_counts, expected_counts),
    )
    daily_truth = dict((truth_control.get("daily_universal_oos") or {}).get("actual") or {})
    check_true(
        "mr13s_truth_geometry_reports_population_and_independence_enrichment",
        int(daily_truth.get("population_n", 0) or 0) == len(frame)
        and dict(daily_truth.get("s5_m5") or {}).get("independence_enrichment") is not None
        and dict(daily_truth.get("s4plus_m4plus") or {}).get("independence_enrichment") is not None,
    )

    project_root = Path(__file__).resolve().parents[2]
    check_true(
        "mr13s_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )
    report_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    check_true(
        "mr13s_joint_geometry_is_model_specific_extension_without_new_audit",
        "Model-specific Extension｜{payload['model_research_id']}｜Multi-head Learnability" in report_source
        and "Model-specific Extension｜{payload['model_research_id']}｜Truth / Prediction Geometry" in report_source
        and "Truth / Prediction Geometry" in report_source
        and "predicted_joint_geometry" in report_source
        and "safety_cohorts" in report_source
        and "markdown_tone" in report_source,
    )
    app_source = (
        project_root / "services" / "research" / "breakout_quality_application.py"
    ).read_text(encoding="utf-8")
    contract_source = (project_root / "core" / "research_report_contract.py").read_text(encoding="utf-8")
    check_true(
        "mr13s_truth_geometry_is_extension_not_standard_section_without_dedicated_menu",
        '"truth_prediction_geometry": ModelExtensionContract(' in contract_source
        and 'S("truth_prediction_geometry"' not in contract_source
        and "Actual MFE×Safety Truth Geometry（只讀）" not in app_source
        and "Pred Safety↔Raw-MFE Daily rho" in app_source,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_shared_ah_contract_case(_base_params):
    """Protect Shared-AH controls: A1, historical A2, A3 target pivot, and AM+A2 composition."""

    case_id = "BREAKOUT_QUALITY_MR13AK_SHARED_AH"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY,
        CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
        get_continuous_ranker_pair_weight_policy,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import (
        _pairwise_logistic_loss,
        _same_date_average_rank_percentile,
        _train_epoch,
    )
    from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    model_spec = get_model_spec(str(profile.model_architecture))

    a2_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    a2_research = get_continuous_ranker_research_spec(a2_profile.name)
    a2_recipe = get_continuous_ranker_execution_recipe(a2_profile.name)
    a2_model_spec = get_model_spec(str(a2_profile.model_architecture))

    a3_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    a3_research = get_continuous_ranker_research_spec(a3_profile.name)
    a3_recipe = get_continuous_ranker_execution_recipe(a3_profile.name)
    a3_model_spec = get_model_spec(str(a3_profile.model_architecture))

    combined_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_FULL_HORIZON_OPPORTUNITY_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    combined_research = get_continuous_ranker_research_spec(combined_profile.name)
    combined_recipe = get_continuous_ranker_execution_recipe(combined_profile.name)
    combined_model_spec = get_model_spec(str(combined_profile.model_architecture))

    check(
        "mr13ak_identity_and_model_gate_authorization",
        (
            "MR-13AK",
            "inception_time_shared_safety_mfe_v1",
            False,
            False,
        ),
        (
            research.model_research_id,
            profile.model_architecture,
            research.selection_pit_authorized,
            research.current_time_validation_authorized,
        ),
    )
    check_true(
        "mr13ak_reuses_daily_safety_raw_mfe_targets_without_external_mr13m_context",
        recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE
        and recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE
        and recipe.pairwise_reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
        and recipe.context_policy.source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
        and recipe.context_policy.roles == ()
        and not bool(recipe.dependency_spec.requires_continuous_target_artifact)
        and recipe.dependency_spec.context_source == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
    )
    check(
        "mr13ak_architecture_is_shared_encoder_with_independent_safety_and_mfe_heads",
        ("inception_time_shared_safety_mfe", ("global_average", "raw_safety_head", "raw_mfe_head"), False),
        (model_spec.family, model_spec.pooling, bool(model_spec.use_dataset_context)),
    )

    sem = dict(training_semantics(profile).get("shared_safety_weighted_mfe_duo_head_contract") or {})
    check_true(
        "mr13ak_semantics_freeze_a1_scientific_control",
        sem.get("safety_target") == "same_date_low_adverse_safety_percentile"
        and sem.get("raw_mfe_target") == "same_date_pure_mfe_percentile"
        and sem.get("architecture") == "shared_encoder_independent_raw_safety_and_raw_mfe_heads"
        and sem.get("mfe_head_inputs") == "shared_latent_only_no_safety_prediction_input"
        and sem.get("mfe_pair_context") == "stop_gradient_same_date_average_rank_percentile_of_raw_safety_probability"
        and sem.get("mfe_pair_safety_weight")
        == "same_date_predicted_safety_percentile_i_times_j"
        and sem.get("mfe_pair_direction") == "pure_mfe_only_never_reversed_by_safety"
        and sem.get("safety_head_gradient_from_mfe_loss") is False
        and sem.get("shared_encoder_gradient_from_both_heads") is True
        and sem.get("head_weighting") == "fixed_equal_mean_no_lambda_sweep"
        and sem.get("external_predicted_safety_dependency") is False
        and sem.get("epoch_selection") == "raw_mfe_mean_daily_spearman"
        and sem.get("runtime_score") == "raw_mfe_pass_probability_only",
    )

    check(
        "mr13al_a2_identity_reuses_existing_safety_context_topology",
        (
            "MR-13AL",
            "inception_time_safety_conditional_mfe_v1",
            ("global_average", "raw_safety_head", "conditional_mfe_head"),
            False,
            False,
        ),
        (
            a2_research.model_research_id,
            a2_profile.model_architecture,
            a2_model_spec.pooling,
            a2_research.selection_pit_authorized,
            a2_research.current_time_validation_authorized,
        ),
    )
    check_true(
        "shared_ah_final_mfe_topology_semantics_are_owned_by_model_spec",
        model_spec.final_mfe_topology_contract()
        == {
            "architecture": "shared_encoder_independent_raw_safety_and_raw_mfe_heads",
            "mfe_head_inputs": "shared_latent_only_no_safety_prediction_input",
        }
        and a2_model_spec.final_mfe_topology_contract()
        == {
            "architecture": "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head",
            "mfe_head_inputs": "shared_latent_plus_stop_gradient_raw_safety_probability",
        },
    )
    training_contract_source = read_source_text(
        "filters/breakout_quality/ranker_training_contract.py"
    )
    check_true(
        "training_contract_consumer_does_not_redecode_final_mfe_head_tokens",
        '"raw_mfe_head" in heads' not in training_contract_source
        and '"conditional_mfe_head" in heads' not in training_contract_source,
    )

    check_true(
        "mr13al_a2_keeps_a1_training_recipe_and_changes_only_model_topology",
        a2_profile.training_objective == profile.training_objective
        and a2_profile.continuous_target_id == profile.continuous_target_id
        and a2_profile.loss_name == profile.loss_name
        and a2_profile.epoch_selection_metric == profile.epoch_selection_metric
        and a2_profile.optimizer_name == profile.optimizer_name
        and a2_profile.training_sampling_mode == profile.training_sampling_mode
        and a2_profile.training_label_scope == profile.training_label_scope
        and a2_profile.training_sample_scope == profile.training_sample_scope
        and a2_recipe.training_policy == recipe.training_policy
        and a2_recipe.pairwise_reduction == recipe.pairwise_reduction
        and a2_recipe.context_policy == recipe.context_policy
        and a2_profile.model_architecture != profile.model_architecture,
    )

    a2_sem = dict(training_semantics(a2_profile).get("shared_safety_weighted_mfe_duo_head_contract") or {})
    check_true(
        "mr13al_a2_artifact_semantics_describe_stop_gradient_safety_context_topology",
        a2_sem.get("architecture") == "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head"
        and a2_sem.get("mfe_head_inputs") == "shared_latent_plus_stop_gradient_raw_safety_probability"
        and a2_sem.get("mfe_pair_context") == sem.get("mfe_pair_context")
        and a2_sem.get("mfe_pair_safety_weight") == sem.get("mfe_pair_safety_weight")
        and a2_sem.get("mfe_pair_direction") == sem.get("mfe_pair_direction")
        and a2_sem.get("head_weighting") == sem.get("head_weighting"),
    )

    a3_sem = dict(training_semantics(a3_profile).get("shared_safety_weighted_primary_duo_head_contract") or {})
    check_true(
        "mr13am_a3_is_ak_target_only_control_without_al_safety_context",
        a3_research.model_research_id == "MR-13AM"
        and a3_profile.model_architecture == profile.model_architecture
        and a3_model_spec.pooling == model_spec.pooling
        and a3_profile.continuous_target_id == "daily_full_horizon_opportunity_r_v1"
        and a3_profile.continuous_target_id != profile.continuous_target_id
        and a3_recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_PRIMARY
        and a3_recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE
        and a3_recipe.pairwise_reduction == recipe.pairwise_reduction
        and a3_recipe.context_policy == recipe.context_policy
        and a3_research.selection_pit_authorized is False
        and a3_research.current_time_validation_authorized is False
        and a3_sem.get("architecture") == sem.get("architecture")
        and a3_sem.get("primary_head_inputs") == "shared_latent_only_no_safety_prediction_input"
        and a3_sem.get("primary_pair_safety_weight")
        == "same_date_predicted_safety_percentile_i_times_j"
        and a3_sem.get("primary_pair_direction")
        == "profile_primary_target_only_never_reversed_by_safety"
        and a3_sem.get("head_weighting") == "fixed_equal_mean_no_lambda_sweep"
        and a3_sem.get("primary_target_id") == "daily_full_horizon_opportunity_r_v1",
    )
    combined_sem = dict(
        training_semantics(combined_profile).get("shared_safety_weighted_primary_duo_head_contract") or {}
    )
    check_true(
        "mr13an_combines_am_economic_target_with_a2_stop_gradient_safety_context_only",
        combined_research.model_research_id == "MR-13AN"
        and combined_profile.training_objective == a3_profile.training_objective
        and combined_profile.continuous_target_id == a3_profile.continuous_target_id
        and combined_profile.loss_name == a3_profile.loss_name
        and combined_profile.epoch_selection_metric == a3_profile.epoch_selection_metric
        and combined_profile.optimizer_name == a3_profile.optimizer_name
        and combined_profile.training_sampling_mode == a3_profile.training_sampling_mode
        and combined_profile.training_label_scope == a3_profile.training_label_scope
        and combined_profile.training_sample_scope == a3_profile.training_sample_scope
        and combined_recipe.training_policy == a3_recipe.training_policy
        and combined_recipe.pairwise_reduction == a3_recipe.pairwise_reduction
        and combined_recipe.context_policy == a3_recipe.context_policy
        and combined_profile.model_architecture == a2_profile.model_architecture
        and combined_profile.model_architecture != a3_profile.model_architecture
        and combined_model_spec.final_mfe_topology_contract()
        == a2_model_spec.final_mfe_topology_contract()
        and combined_sem.get("architecture")
        == "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head"
        and combined_sem.get("primary_head_inputs")
        == "shared_latent_plus_stop_gradient_raw_safety_probability"
        and combined_sem.get("primary_target_id") == "daily_full_horizon_opportunity_r_v1"
        and combined_sem.get("primary_pair_safety_weight")
        == a3_sem.get("primary_pair_safety_weight")
        and combined_sem.get("primary_pair_direction") == a3_sem.get("primary_pair_direction")
        and combined_sem.get("head_weighting") == a3_sem.get("head_weighting")
        and combined_research.selection_pit_authorized is False
        and combined_research.current_time_validation_authorized is False,
    )

    comparison_pairs = [
        (str(model_id), str(profile_name))
        for model_id, profile_name in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    ]
    current_training_pair = tuple(
        str(value) for value in breakout_quality_config.BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    check_true(
        "continuous_dl_comparison_membership_keeps_ak_reference_and_injects_current_training_identity",
        current_training_pair in comparison_pairs
        and any(model_id == "MR-13AK" for model_id, _profile_name in comparison_pairs)
        and "MR-13AL" not in {model_id for model_id, _profile_name in comparison_pairs}
        and "MR-13AN" not in {model_id for model_id, _profile_name in comparison_pairs},
    )

    torch.manual_seed(42)
    model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_shared_safety_mfe_v1",
    )
    x = torch.randn(6, 300, 10)
    context = torch.empty((6, 0), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        safety_logits, mfe_before = model.forward_safety_mfe_heads(x, context)
        both = model.forward_output_head(x, context, "both")
        final_logits = model.forward_output_head(x, context, "final")
        model.raw_safety_classifier.weight.add_(3.0)
        model.raw_safety_classifier.bias.sub_(2.0)
        _changed_safety, mfe_after = model.forward_safety_mfe_heads(x, context)
    check_true(
        "mr13ak_mfe_head_does_not_take_safety_prediction_as_input",
        tuple(safety_logits.shape) == (6, 2)
        and tuple(mfe_before.shape) == (6, 2)
        and tuple(both.shape) == (6, 4)
        and torch.equal(final_logits, mfe_before)
        and torch.equal(mfe_before, mfe_after),
    )

    torch.manual_seed(42)
    a2_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_safety_conditional_mfe_v1",
    )
    a2_model.eval()
    with torch.no_grad():
        a2_safety_before, a2_mfe_before = a2_model.forward_safety_mfe_heads(x, context)
        a2_model.raw_safety_classifier.weight.add_(3.0)
        a2_model.raw_safety_classifier.bias.sub_(2.0)
        a2_safety_after, a2_mfe_after = a2_model.forward_safety_mfe_heads(x, context)
    check_true(
        "mr13al_a2_mfe_head_explicitly_consumes_detached_safety_context",
        not torch.equal(a2_safety_before, a2_safety_after)
        and not torch.equal(a2_mfe_before, a2_mfe_after),
    )

    torch.manual_seed(42)
    a2_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_safety_conditional_mfe_v1",
    )
    a2_model.train()
    _a2_safety_logits, a2_mfe_logits = a2_model.forward_safety_mfe_heads(x, context)
    a2_model.zero_grad(set_to_none=True)
    a2_mfe_logits.float().sum().backward()
    a2_safety_grad = a2_model.raw_safety_classifier.weight.grad
    a2_mfe_grad = a2_model.conditional_mfe_classifier.weight.grad
    a2_encoder_grad = next(a2_model.inception_modules[0].parameters()).grad
    check_true(
        "mr13al_a2_safety_context_is_stop_gradient_but_mfe_updates_encoder_and_final_head",
        a2_safety_grad is None
        and a2_mfe_grad is not None
        and float(a2_mfe_grad.abs().sum().item()) > 0.0
        and a2_encoder_grad is not None
        and float(a2_encoder_grad.abs().sum().item()) > 0.0,
    )

    # Restore a deterministic fresh model and isolate the MFE loss. Detached Safety
    # probabilities may weight the MFE pairs but must not send gradient to Safety head.
    torch.manual_seed(42)
    model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_shared_safety_mfe_v1",
    )
    model.train()
    safety_logits, mfe_logits = model.forward_safety_mfe_heads(x, context)
    safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
    mfe_margin = mfe_logits.float()[:, 1] - mfe_logits.float()[:, 0]
    pure_mfe_target = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float32)
    dates = pd.to_datetime(["2026-01-05"] * 6)
    safety_percentile = _same_date_average_rank_percentile(torch, safety_probability, dates)
    weighted_target = torch.stack([pure_mfe_target, safety_percentile], dim=1)
    mfe_loss, pair_count = _pairwise_logistic_loss(
        torch,
        mfe_margin,
        weighted_target,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
    )
    check_true("mr13ak_weighted_mfe_fixture_has_rankable_pairs", mfe_loss is not None and pair_count == 15)
    model.zero_grad(set_to_none=True)
    mfe_loss.backward()
    safety_grad = model.raw_safety_classifier.weight.grad
    mfe_grad = model.raw_mfe_classifier.weight.grad
    encoder_grad = next(model.inception_modules[0].parameters()).grad
    check_true(
        "mr13ak_mfe_loss_stop_gradient_isolates_safety_classifier_but_not_shared_encoder",
        safety_grad is None
        and mfe_grad is not None
        and bool(torch.isfinite(mfe_grad).all())
        and float(mfe_grad.abs().sum().item()) > 0.0
        and encoder_grad is not None
        and bool(torch.isfinite(encoder_grad).all())
        and float(encoder_grad.abs().sum().item()) > 0.0,
    )

    # Exercise the actual A1 training handler on CPU so the executable branch—not
    # only the primitive arithmetic—must produce a finite update.
    torch.manual_seed(42)
    epoch_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_shared_safety_mfe_v1",
    )
    epoch_plan = resolve_torch_execution_plan(
        torch,
        requested_device="cpu",
        mixed_precision=False,
        mixed_precision_dtype="auto",
        deterministic_algorithms=True,
        allow_tf32=False,
    )
    epoch_model = epoch_model.to(epoch_plan.device)
    epoch_optimizer = torch.optim.Adam(epoch_model.parameters(), lr=1e-4)
    epoch_feature_bank = np.random.default_rng(42).normal(size=(6, 300, 10)).astype(np.float32)
    epoch_context = np.empty((6, 0), dtype=np.float32)
    epoch_target = np.stack([
        np.asarray([0.1, 0.8, 0.4, 0.9, 0.2, 0.7], dtype=np.float32),
        np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32),
    ], axis=1)
    before = epoch_model.raw_mfe_classifier.weight.detach().clone()
    epoch_loss = _train_epoch(
        torch,
        epoch_model,
        epoch_optimizer,
        epoch_feature_bank,
        epoch_context,
        np.arange(6, dtype=np.int64),
        epoch_target,
        pd.to_datetime(["2026-01-05"] * 6),
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        batch_size=128,
        seed=42,
        gradient_clip_norm=1.0,
        plan=epoch_plan,
        grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    after = epoch_model.raw_mfe_classifier.weight.detach()
    check_true(
        "shared_safety_weighted_mfe_training_handler_executes_finite_dual_head_update",
        np.isfinite(float(epoch_loss)) and not torch.equal(before, after),
    )

    torch.manual_seed(42)
    a2_epoch_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_safety_conditional_mfe_v1",
    ).to(epoch_plan.device)
    a2_optimizer = torch.optim.Adam(a2_epoch_model.parameters(), lr=1e-4)
    a2_before = a2_epoch_model.conditional_mfe_classifier.weight.detach().clone()
    a2_epoch_loss = _train_epoch(
        torch,
        a2_epoch_model,
        a2_optimizer,
        epoch_feature_bank,
        epoch_context,
        np.arange(6, dtype=np.int64),
        epoch_target,
        pd.to_datetime(["2026-01-05"] * 6),
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
        batch_size=128,
        seed=42,
        gradient_clip_norm=1.0,
        plan=epoch_plan,
        grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    a2_after = a2_epoch_model.conditional_mfe_classifier.weight.detach()
    check_true(
        "shared_safety_weighted_loss_handler_is_topology_agnostic_for_a2_context_head",
        np.isfinite(float(a2_epoch_loss)) and not torch.equal(a2_before, a2_after),
    )

    # A3 reuses the exact A1 topology/loss engine. Only the second target column
    # changes to the profile primary target, so a finite update must occur without
    # introducing a new trainer branch or Safety-context input.
    torch.manual_seed(42)
    a3_epoch_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_shared_safety_mfe_v1",
    ).to(epoch_plan.device)
    a3_optimizer = torch.optim.Adam(a3_epoch_model.parameters(), lr=1e-4)
    a3_before = a3_epoch_model.raw_mfe_classifier.weight.detach().clone()
    a3_epoch_target = np.stack([
        np.asarray([0.1, 0.8, 0.4, 0.9, 0.2, 0.7], dtype=np.float32),
        np.asarray([0.8, 0.1, 0.7, 0.2, 0.6, 0.3], dtype=np.float32),
    ], axis=1)
    a3_epoch_loss = _train_epoch(
        torch,
        a3_epoch_model,
        a3_optimizer,
        epoch_feature_bank,
        epoch_context,
        np.arange(6, dtype=np.int64),
        a3_epoch_target,
        pd.to_datetime(["2026-01-05"] * 6),
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        batch_size=128,
        seed=42,
        gradient_clip_norm=1.0,
        plan=epoch_plan,
        grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    a3_after = a3_epoch_model.raw_mfe_classifier.weight.detach()
    check_true(
        "mr13am_a3_reuses_shared_safety_weighted_engine_with_primary_target_direction",
        np.isfinite(float(a3_epoch_loss)) and not torch.equal(a3_before, a3_after),
    )

    # MR-13AN composes the same primary-target objective with the already-supported
    # Safety-conditioned final head. The generic trainer must handle this composition
    # without a new objective/architecture branch.
    torch.manual_seed(42)
    combined_epoch_model = build_active_model(
        feature_count=10,
        context_count=0,
        architecture="inception_time_safety_conditional_mfe_v1",
    ).to(epoch_plan.device)
    combined_optimizer = torch.optim.Adam(combined_epoch_model.parameters(), lr=1e-4)
    combined_before = combined_epoch_model.conditional_mfe_classifier.weight.detach().clone()
    combined_epoch_loss = _train_epoch(
        torch,
        combined_epoch_model,
        combined_optimizer,
        epoch_feature_bank,
        epoch_context,
        np.arange(6, dtype=np.int64),
        a3_epoch_target,
        pd.to_datetime(["2026-01-05"] * 6),
        training_objective=TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_PRIMARY_PAIRWISE_RANKING,
        batch_size=128,
        seed=42,
        gradient_clip_norm=1.0,
        plan=epoch_plan,
        grad_scaler=None,
        pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    combined_after = combined_epoch_model.conditional_mfe_classifier.weight.detach()
    check_true(
        "mr13an_primary_target_plus_safety_context_uses_existing_generic_loss_engine",
        np.isfinite(float(combined_epoch_loss))
        and not torch.equal(combined_before, combined_after),
    )

    percentile_fixture = _same_date_average_rank_percentile(
        torch,
        torch.tensor([0.2, 0.8, 0.8, 0.9, 0.1], dtype=torch.float32),
        pd.to_datetime(["2026-01-05"] * 3 + ["2026-01-06"] * 2),
    )
    check_true(
        "mr13ak_shared_safety_preserves_ah_same_date_average_rank_percentile_scale",
        torch.allclose(
            percentile_fixture,
            torch.tensor([0.0, 0.75, 0.75, 1.0, 0.0], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        ),
    )

    pair_policy = get_continuous_ranker_pair_weight_policy(
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY
    )
    left = torch.tensor([0.2, 0.8], dtype=torch.float32)
    right = torch.tensor([0.5, 0.4], dtype=torch.float32)
    actual_product = pair_policy.apply(torch, torch.ones(2), left, right)
    check_true(
        "mr13ak_ah_pair_multiplier_is_exact_s_i_times_s_j",
        torch.allclose(actual_product, left * right, rtol=0.0, atol=0.0),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_safety_raw_mfe_hmhs_tri_head_contract_case(_base_params):
    """Protect MR-13T as a raw-input direct-HM/HS supervision contrast to MR-13S."""

    case_id = "BREAKOUT_QUALITY_SAFETY_RAW_MFE_HMHS_TRI_HEAD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED,
        get_continuous_ranker_training_policy,
    )
    from filters.breakout_quality.conditional_mfe_opportunity import (
        HMHS_HIGH_PERCENTILE_CUTOFF,
        build_conditional_mfe_opportunity_targets,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.inference import strict_parallel_batched_logits
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.ranker_training import safety_raw_mfe_hmhs_metrics
    from services.breakout_quality.train_continuous_ranker import _combine_training_head_losses

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_DUO_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13t_historical_raw_input_model_only_controlled_contrast_remains_registered",
        (
            "MR-13T",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
            "inception_time_safety_raw_mfe_hmhs_v1",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check(
        "mr13t_keeps_mr13s_raw_mfe_epoch_selection_and_pairwise_reduction",
        (
            control.epoch_selection_metric,
            get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
        ),
        (
            profile.epoch_selection_metric,
            get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction,
        ),
    )

    control_model_spec = get_model_spec(control.model_architecture)
    tri_model_spec = get_model_spec(profile.model_architecture)
    check_true(
        "mr13t_changes_head_structure_without_artificial_input_representation",
        len(FEATURE_COLUMNS) == 10
        and tuple(control_model_spec.sequence_input_paths) == ("raw_level",)
        and tuple(tri_model_spec.sequence_input_paths) == ("raw_level",)
        and not bool(control_model_spec.use_dataset_context)
        and not bool(tri_model_spec.use_dataset_context)
        and (
            control_model_spec.inception_depth,
            control_model_spec.inception_filters,
            control_model_spec.inception_bottleneck_channels,
            tuple(control_model_spec.inception_kernel_sizes or ()),
            control_model_spec.inception_residual_every,
            control_model_spec.normalization,
            control_model_spec.dropout,
        )
        == (
            tri_model_spec.inception_depth,
            tri_model_spec.inception_filters,
            tri_model_spec.inception_bottleneck_channels,
            tuple(tri_model_spec.inception_kernel_sizes or ()),
            tri_model_spec.inception_residual_every,
            tri_model_spec.normalization,
            tri_model_spec.dropout,
        ),
    )

    sem = dict(training_semantics(profile).get("safety_raw_mfe_hmhs_tri_head_contract") or {})
    check(
        "mr13t_contract_is_direct_hmhs_shared_latent_three_head_equal_loss",
        (
            "same_date_low_adverse_safety_percentile",
            "same_date_pure_mfe_percentile",
            "indicator_of_safety_percentile_ge_0.5_and_pure_mfe_percentile_ge_0.5",
            "stop_gradient_raw_safety_probability_for_raw_mfe_head_only",
            "shared_raw_latent_only_no_safety_or_mfe_score_arithmetic",
            "each_head_loss_updates_own_classifier_only",
            True,
            "fixed_equal_mean_three_heads_no_lambda_sweep",
            "raw_mfe_mean_daily_spearman_same_as_mr13s_control",
            "model_gate_only_no_pit_no_strategy_conversion",
        ),
        (
            sem.get("safety_target"),
            sem.get("raw_mfe_target"),
            sem.get("joint_hmhs_target"),
            sem.get("conditional_context"),
            sem.get("joint_head_inputs"),
            sem.get("classifier_isolation"),
            sem.get("shared_encoder_gradient_from_all_heads"),
            sem.get("head_weighting"),
            sem.get("epoch_selection"),
            sem.get("runtime_status"),
        ),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 10 + ["2026-01-05"] * 10),
            "target_favorable_r": list(range(1, 11)) * 2,
            "target_adverse_r": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 1.0, 0.5, 0.6, 0.1] * 2,
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
        }
    )
    targets = build_conditional_mfe_opportunity_targets(
        frame, np.ones(len(frame), dtype=bool)
    )
    tri_target = targets.safety_raw_mfe_hmhs_training_target
    expected_hmhs = np.asarray(
        (targets.low_adverse_safety_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF)
        & (targets.primary_mfe_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF),
        dtype=np.float32,
    )
    check("mr13t_training_target_shape", (20, 3), tri_target.shape)
    check_true(
        "mr13t_first_two_targets_are_byte_identical_to_mr13s_and_joint_is_exact_intersection",
        bool(
            np.array_equal(tri_target[:, :2], targets.safety_raw_mfe_training_target)
            and np.array_equal(tri_target[:, 2], expected_hmhs)
            and np.array_equal(targets.direct_hmhs_target, expected_hmhs)
        ),
    )

    torch, _nn = require_torch()
    torch.manual_seed(17)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=profile.model_architecture,
    )
    x = torch.randn(4, 300, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    safety_logits, mfe_logits, joint_logits = model.forward_safety_raw_mfe_hmhs_heads(x, context)
    check(
        "mr13t_model_exposes_three_two_logit_heads_and_keeps_raw_mfe_as_default_score",
        ((4, 2), (4, 2), (4, 2), (4, 2)),
        (
            tuple(safety_logits.shape),
            tuple(mfe_logits.shape),
            tuple(joint_logits.shape),
            tuple(model(x, context).shape),
        ),
    )
    class _TriHeadInferenceProbe(_nn.Module):
        def forward_output_head(self, probe_x, probe_context, output_head):
            if str(output_head) != "tri_head":
                raise ValueError("probe只接受tri_head")
            return torch.zeros((probe_x.shape[0], 6), dtype=probe_x.dtype)

    probe_features = np.zeros((4, 2, 1), dtype=np.float32)
    probe_context = np.empty((4, 0), dtype=np.float32)
    tri_logits = strict_parallel_batched_logits(
        torch,
        _TriHeadInferenceProbe(),
        probe_features,
        probe_context,
        indices=np.arange(len(probe_features), dtype=np.int64),
        batch_size=2,
        workers=1,
        execution_plan=None,
        output_head="tri_head",
    )
    check(
        "mr13t_shared_batched_inference_allocates_six_logit_tri_head_buffer",
        (4, 6),
        tuple(tri_logits.shape),
    )

    def grad_present(parameter) -> bool:
        return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())

    shared_parameter = next(model.inception_modules[0].parameters())
    model.zero_grad(set_to_none=True)
    model.forward_safety_raw_mfe_hmhs_heads(x, context)[2].sum().backward()
    check_true(
        "mr13t_joint_loss_updates_joint_and_shared_only",
        grad_present(model.joint_hmhs_classifier.weight)
        and grad_present(shared_parameter)
        and model.raw_safety_classifier.weight.grad is None
        and model.conditional_mfe_classifier.weight.grad is None,
    )
    model.zero_grad(set_to_none=True)
    model.forward_safety_raw_mfe_hmhs_heads(x, context)[1].sum().backward()
    check_true(
        "mr13t_raw_mfe_loss_keeps_safety_and_joint_classifiers_isolated",
        grad_present(model.conditional_mfe_classifier.weight)
        and grad_present(shared_parameter)
        and model.raw_safety_classifier.weight.grad is None
        and model.joint_hmhs_classifier.weight.grad is None,
    )
    model.zero_grad(set_to_none=True)
    model.forward_safety_raw_mfe_hmhs_heads(x, context)[0].sum().backward()
    check_true(
        "mr13t_safety_loss_keeps_mfe_and_joint_classifiers_isolated",
        grad_present(model.raw_safety_classifier.weight)
        and grad_present(shared_parameter)
        and model.conditional_mfe_classifier.weight.grad is None
        and model.joint_hmhs_classifier.weight.grad is None,
    )

    mirror_safety = targets.low_adverse_safety_percentile.astype(np.float32)
    mirror_mfe = (1.0 - targets.low_adverse_safety_percentile).astype(np.float32)
    direct_score = expected_hmhs.astype(np.float32)
    metrics = safety_raw_mfe_hmhs_metrics(
        np.arange(len(frame), dtype=np.int64),
        frame,
        targets,
        {
            "raw_safety": mirror_safety,
            "raw_mfe": mirror_mfe,
            "joint_hmhs": direct_score,
        },
    )
    joint = dict(metrics.get("joint_hmhs") or {})
    product = dict(metrics.get("joint_product_control") or {})
    check_true(
        "mr13t_direct_joint_metrics_can_detect_information_missed_by_marginal_product",
        joint.get("pairwise_concordance") is not None
        and product.get("pairwise_concordance") is not None
        and float(joint["pairwise_concordance"]) > float(product["pairwise_concordance"])
        and float(dict(joint.get("top_10pct") or {}).get("hmhs_enrichment") or 0.0)
        > float(dict(product.get("top_10pct") or {}).get("hmhs_enrichment") or 0.0),
    )

    project_root = Path(__file__).resolve().parents[2]
    training_policy = get_continuous_ranker_training_policy(profile.training_objective)
    combined_probe_loss = _combine_training_head_losses(
        [
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(2.0, dtype=torch.float32),
            torch.tensor(6.0, dtype=torch.float32),
        ],
        combination=training_policy.head_loss_combination,
        component_count=training_policy.head_loss_component_count,
    )
    check_true(
        "mr13t_training_uses_fixed_equal_three_head_mean_and_raw_mfe_epoch_selection",
        training_policy.head_loss_combination
        == CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_REQUIRED
        and int(training_policy.head_loss_component_count) == 3
        and torch.allclose(
            combined_probe_loss,
            torch.tensor(3.0, dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )
        and profile.epoch_selection_metric == "raw_mfe_mean_daily_spearman",
    )
    daily_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    app_source = (
        project_root / "services" / "research" / "breakout_quality_application.py"
    ).read_text(encoding="utf-8")
    check_true(
        "mr13t_joint_result_is_model_specific_extension_and_oos_artifact_without_standard_sop_pollution",
        'oos_frame["joint_hmhs_score"]' in daily_source
        and 'oos_frame["target_direct_hmhs"]' in daily_source
        and "Direct HM/HS Joint Retrieval" in daily_source
        and "Model-specific Extension" in daily_source
        and "Model-specific Extension" in app_source
        and "標準模型 SOP｜3. Direct HM/HS Joint Retrieval" not in app_source,
    )
    check_true(
        "mr13t_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_hmhs_single_head_contract_case(_base_params):
    """Protect MR-13U as the raw-input Direct-HM/HS H-only learnability ablation."""

    case_id = "BREAKOUT_QUALITY_HMHS_SINGLE_HEAD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_opportunity import (
        HMHS_HIGH_PERCENTILE_CUTOFF,
        build_conditional_mfe_opportunity_targets,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.ranker_training import direct_hmhs_metrics

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_HMHS_SINGLE_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13u_historical_h_only_model_gate_profile_remains_registered",
        (
            "MR-13U",
            TRAINING_OBJECTIVE_DAILY_HMHS_PAIRWISE_RANKING,
            "inception_time_v1",
            "hmhs_pairwise_concordance",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture or "inception_time_v1",
            profile.epoch_selection_metric,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check(
        "mr13u_keeps_mr13t_full_list_pairwise_reduction",
        get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
        get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction,
    )

    h_spec = get_model_spec(profile.model_architecture)
    tri_spec = get_model_spec(control.model_architecture)
    check_true(
        "mr13u_keeps_raw_300x10_inceptiontime_trunk_without_artificial_inputs",
        len(FEATURE_COLUMNS) == 10
        and tuple(h_spec.sequence_input_paths) == ("raw_level",)
        and tuple(tri_spec.sequence_input_paths) == ("raw_level",)
        and not bool(h_spec.use_dataset_context)
        and not bool(tri_spec.use_dataset_context)
        and (
            h_spec.inception_depth,
            h_spec.inception_filters,
            h_spec.inception_bottleneck_channels,
            tuple(h_spec.inception_kernel_sizes or ()),
            h_spec.inception_residual_every,
            h_spec.normalization,
            h_spec.dropout,
        )
        == (
            tri_spec.inception_depth,
            tri_spec.inception_filters,
            tri_spec.inception_bottleneck_channels,
            tuple(tri_spec.inception_kernel_sizes or ()),
            tri_spec.inception_residual_every,
            tri_spec.normalization,
            tri_spec.dropout,
        ),
    )

    sem = dict(training_semantics(profile).get("direct_hmhs_single_head_contract") or {})
    check(
        "mr13u_contract_is_h_only_full_list_model_gate",
        (
            "indicator_of_safety_percentile_ge_0.5_and_pure_mfe_percentile_ge_0.5",
            "single_head_inception_time_raw_300x10",
            "shared_raw_latent_only",
            "raw_safety_and_raw_mfe",
            "full_list_delta_ndcg_pairwise_logistic",
            "validation_hmhs_pairwise_concordance_then_global_pr_auc_tiebreak",
            "model_gate_only_no_pit_no_strategy_conversion",
        ),
        (
            sem.get("joint_hmhs_target"),
            sem.get("architecture"),
            sem.get("joint_head_inputs"),
            sem.get("removed_auxiliary_heads"),
            sem.get("head_losses"),
            sem.get("epoch_selection"),
            sem.get("runtime_status"),
        ),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 10 + ["2026-01-05"] * 10),
            "target_favorable_r": list(range(1, 11)) * 2,
            "target_adverse_r": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 1.0, 0.5, 0.6, 0.1] * 2,
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
        }
    )
    targets = build_conditional_mfe_opportunity_targets(frame, np.ones(len(frame), dtype=bool))
    expected = np.asarray(
        (targets.low_adverse_safety_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF)
        & (targets.primary_mfe_percentile >= HMHS_HIGH_PERCENTILE_CUTOFF),
        dtype=np.float32,
    )
    check_true(
        "mr13u_target_is_exact_canonical_hmhs_intersection",
        np.array_equal(targets.direct_hmhs_target, expected),
    )

    torch, _nn = require_torch()
    torch.manual_seed(23)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=profile.model_architecture,
    )
    x = torch.randn(4, 300, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    logits = model(x, context)
    check("mr13u_model_is_single_two_logit_head", (4, 2), tuple(logits.shape))
    check_true(
        "mr13u_model_has_no_mr13t_auxiliary_classifiers",
        getattr(model, "raw_safety_classifier", None) is None
        and getattr(model, "conditional_mfe_classifier", None) is None
        and getattr(model, "joint_hmhs_classifier", None) is None,
    )

    direct_score = expected.astype(np.float32)
    metrics = direct_hmhs_metrics(
        np.arange(len(frame), dtype=np.int64), frame, targets, direct_score
    )
    check_true(
        "mr13u_direct_hmhs_metric_surface_rewards_perfect_h_ranking",
        float(metrics.get("pairwise_concordance") or 0.0) >= 0.999
        and float(metrics.get("global_average_precision") or 0.0) >= 0.999
        and float(dict(metrics.get("top_10pct") or {}).get("hmhs_enrichment") or 0.0) > 1.0,
    )

    project_root = Path(__file__).resolve().parents[2]
    trainer_source = (project_root / "services" / "breakout_quality" / "train_continuous_ranker.py").read_text(encoding="utf-8")
    daily_source = (project_root / "services" / "breakout_quality" / "train_daily_ranker.py").read_text(encoding="utf-8")
    app_source = (project_root / "services" / "research" / "breakout_quality_application.py").read_text(encoding="utf-8")
    check_true(
        "mr13u_epoch_selection_is_validation_h_pair_then_pr_auc_without_oos_fit",
        "best_hmhs_pairwise_concordance" in trainer_source
        and "best_hmhs_global_average_precision" in trainer_source
        and "Val HM/HS Pair" in trainer_source
        and profile.epoch_selection_metric == "hmhs_pairwise_concordance",
    )
    check_true(
        "mr13u_h_only_evidence_is_model_specific_extension_without_standard_generalization_override",
        'oos_frame["target_direct_hmhs"]' in daily_source
        and "Direct HM/HS H-only Learnability" in daily_source
        and "Model-specific Extension" in daily_source
        and "Model-specific Extension" in app_source
        and "標準模型 SOP｜3. Direct HM/HS H-only Learnability" not in app_source
        and '"Δ HM/HS Pair", "Δ PR-AUC", "Δ Top10×"' not in app_source,
    )
    check_true(
        "mr13u_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_nonlinear_hmhs_head_contract_case(_base_params):
    """Protect MR-13V as the single-mechanism nonlinear Direct-HM/HS readout contrast."""

    case_id = "BREAKOUT_QUALITY_NONLINEAR_HMHS_HEAD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_TRI_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13v_historical_mr13t_nonlinear_head_only_contrast_remains_registered",
        (
            "MR-13V",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
            "inception_time_safety_raw_mfe_hmhs_mlp_v1",
            "raw_mfe_mean_daily_spearman",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            profile.epoch_selection_metric,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check(
        "mr13v_keeps_mr13t_objective_loss_epoch_selection_and_pairwise_reduction",
        (
            control.training_objective,
            control.loss_name,
            control.epoch_selection_metric,
            get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
        ),
        (
            profile.training_objective,
            profile.loss_name,
            profile.epoch_selection_metric,
            get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction,
        ),
    )

    control_spec = get_model_spec(control.model_architecture)
    nonlinear_spec = get_model_spec(profile.model_architecture)
    control_trunk = (
        control_spec.inception_depth,
        control_spec.inception_filters,
        control_spec.inception_bottleneck_channels,
        tuple(control_spec.inception_kernel_sizes or ()),
        control_spec.inception_residual_every,
        control_spec.normalization,
        control_spec.dropout,
        tuple(control_spec.sequence_input_paths),
        bool(control_spec.use_dataset_context),
    )
    nonlinear_trunk = (
        nonlinear_spec.inception_depth,
        nonlinear_spec.inception_filters,
        nonlinear_spec.inception_bottleneck_channels,
        tuple(nonlinear_spec.inception_kernel_sizes or ()),
        nonlinear_spec.inception_residual_every,
        nonlinear_spec.normalization,
        nonlinear_spec.dropout,
        tuple(nonlinear_spec.sequence_input_paths),
        bool(nonlinear_spec.use_dataset_context),
    )
    latent_width = int(nonlinear_spec.inception_filters) * (
        len(tuple(nonlinear_spec.inception_kernel_sizes or ())) + 1
    )
    check_true(
        "mr13v_keeps_raw_300x10_and_mr13t_trunk_while_only_changing_joint_readout",
        int(BREAKOUT_QUALITY_FEATURE_WINDOW_BARS) == 300
        and len(FEATURE_COLUMNS) == 10
        and control_trunk == nonlinear_trunk
        and int(nonlinear_spec.head_width or 0) == latent_width
        and "direct_hmhs_head" in tuple(control_spec.pooling)
        and "direct_hmhs_mlp_head" in tuple(nonlinear_spec.pooling),
    )

    sem = dict(training_semantics(profile).get("safety_raw_mfe_hmhs_tri_head_contract") or {})
    check_true(
        "mr13v_reuses_mr13t_three_head_training_semantics_without_weight_or_target_change",
        sem.get("joint_hmhs_target")
        == "indicator_of_safety_percentile_ge_0.5_and_pure_mfe_percentile_ge_0.5"
        and sem.get("head_weighting") == "fixed_equal_mean_three_heads_no_lambda_sweep"
        and sem.get("epoch_selection") == "raw_mfe_mean_daily_spearman_same_as_mr13s_control"
        and sem.get("joint_head_inputs") == "shared_raw_latent_only_no_safety_or_mfe_score_arithmetic",
    )

    torch, nn = require_torch()
    torch.manual_seed(29)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=profile.model_architecture,
    )
    check_true(
        "mr13v_direct_hmhs_head_is_fixed_one_hidden_layer_relu_mlp",
        isinstance(model.joint_hmhs_classifier, nn.Sequential)
        and len(model.joint_hmhs_classifier) == 3
        and isinstance(model.joint_hmhs_classifier[0], nn.Linear)
        and int(model.joint_hmhs_classifier[0].in_features) == latent_width
        and int(model.joint_hmhs_classifier[0].out_features) == latent_width
        and isinstance(model.joint_hmhs_classifier[1], nn.ReLU)
        and isinstance(model.joint_hmhs_classifier[2], nn.Linear)
        and int(model.joint_hmhs_classifier[2].in_features) == latent_width
        and int(model.joint_hmhs_classifier[2].out_features) == 2,
    )
    x = torch.randn(2, 32, len(FEATURE_COLUMNS))
    context = torch.empty(2, 0)
    safety_logits, mfe_logits, joint_logits = model.forward_safety_raw_mfe_hmhs_heads(x, context)
    check(
        "mr13v_preserves_three_two_logit_output_surface",
        ((2, 2), (2, 2), (2, 2), (2, 6)),
        (
            tuple(safety_logits.shape),
            tuple(mfe_logits.shape),
            tuple(joint_logits.shape),
            tuple(model.forward_output_head(x, context, "tri_head").shape),
        ),
    )

    shared_parameter = next(model.inception_modules[0].parameters())
    model.zero_grad(set_to_none=True)
    model.forward_safety_raw_mfe_hmhs_heads(x, context)[2].sum().backward()
    joint_params = tuple(model.joint_hmhs_classifier.parameters())
    check_true(
        "mr13v_joint_loss_updates_nonlinear_joint_head_and_shared_trunk_only",
        all(parameter.grad is not None for parameter in joint_params)
        and shared_parameter.grad is not None
        and model.raw_safety_classifier.weight.grad is None
        and model.conditional_mfe_classifier.weight.grad is None,
    )

    check_true(
        "mr13v_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_joint_min_target_contract_case(_base_params):
    """Protect MR-13W as the MR-13V-architecture target-only Joint-Min contrast."""

    case_id = "BREAKOUT_QUALITY_JOINT_MIN_TARGET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_opportunity import (
        build_conditional_mfe_opportunity_targets,
    )
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.ranker_training import safety_raw_mfe_joint_min_metrics

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_HMHS_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13w_historical_mr13v_architecture_target_only_contrast_remains_registered",
        (
            "MR-13W",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
            "inception_time_safety_raw_mfe_hmhs_mlp_v1",
            "raw_mfe_mean_daily_spearman",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            profile.epoch_selection_metric,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check_true(
        "mr13w_changes_target_only_not_architecture_loss_epoch_or_pairwise_reduction",
        control.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_HMHS_PAIRWISE_RANKING
        and profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
        and control.model_architecture == profile.model_architecture
        and control.loss_name == profile.loss_name
        and control.epoch_selection_metric == profile.epoch_selection_metric
        and get_continuous_ranker_execution_recipe(control.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        and get_model_spec(control.model_architecture).as_manifest_payload()
        == get_model_spec(profile.model_architecture).as_manifest_payload(),
    )

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"] * 10 + ["2026-01-05"] * 10),
            "target_favorable_r": list(range(1, 11)) * 2,
            "target_adverse_r": [0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 1.0, 0.5, 0.6, 0.1] * 2,
            "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 2,
        }
    )
    targets = build_conditional_mfe_opportunity_targets(
        frame, np.ones(len(frame), dtype=bool)
    )
    expected_min = np.minimum(
        targets.low_adverse_safety_percentile,
        targets.primary_mfe_percentile,
    ).astype(np.float32)
    check_true(
        "mr13w_joint_min_target_is_exact_unthresholded_min_and_preserves_marginal_targets",
        np.array_equal(targets.joint_min_target, expected_min)
        and np.array_equal(
            targets.safety_raw_mfe_joint_min_training_target[:, :2],
            targets.safety_raw_mfe_hmhs_training_target[:, :2],
        )
        and np.array_equal(
            targets.safety_raw_mfe_joint_min_training_target[:, 2], expected_min
        )
        and bool(np.all((expected_min >= 0.0) & (expected_min <= 1.0))),
    )
    # Two HM/HS positives need not be equivalent under Joint-Min. This is the
    # exact information-loss problem the experiment is designed to test.
    hmhs = targets.direct_hmhs_target.astype(bool)
    hmhs_values = np.unique(np.round(expected_min[hmhs], 6))
    check_true(
        "mr13w_continuous_target_retains_depth_inside_binary_hmhs_positive_region",
        bool(hmhs.sum() >= 2 and len(hmhs_values) >= 2),
    )

    sem = dict(
        training_semantics(profile).get("safety_raw_mfe_joint_min_tri_head_contract")
        or {}
    )
    check(
        "mr13w_training_contract_is_parameter_free_continuous_joint_min",
        (
            "same_date_low_adverse_safety_percentile",
            "same_date_pure_mfe_percentile",
            "min_same_date_safety_and_pure_mfe_percentiles",
            "fixed_equal_mean_three_heads_no_lambda_sweep",
            "raw_mfe_mean_daily_spearman_same_as_mr13v_control",
            "model_gate_only_no_pit_no_strategy_conversion",
        ),
        (
            sem.get("safety_target"),
            sem.get("raw_mfe_target"),
            sem.get("joint_min_target"),
            sem.get("head_weighting"),
            sem.get("epoch_selection"),
            sem.get("runtime_status"),
        ),
    )

    perfect_scores = {
        "raw_safety": targets.low_adverse_safety_percentile.copy(),
        "raw_mfe": targets.primary_mfe_percentile.copy(),
        "joint_min": expected_min.copy(),
    }
    metrics = safety_raw_mfe_joint_min_metrics(
        np.arange(len(frame), dtype=np.int64), frame, targets, perfect_scores
    )
    joint = dict(metrics.get("joint_min") or {})
    top10 = dict(joint.get("top_10pct") or {})
    check_true(
        "mr13w_joint_min_metric_surface_measures_rank_depth_and_hmhs_tail_conversion",
        float(joint.get("mean_daily_spearman") or 0.0) >= 0.999
        and float(joint.get("pairwise_concordance") or 0.0) >= 0.999
        and float(top10.get("mean_joint_min") or 0.0)
        > float(joint.get("population_joint_min_mean") or 0.0)
        and float(top10.get("mean_safety") or 0.0) > 0.5
        and float(top10.get("mean_mfe") or 0.0) > 0.5,
    )

    project_root = Path(__file__).resolve().parents[2]
    daily_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    check_true(
        "mr13w_oos_artifact_preserves_continuous_joint_truth_and_score",
        'oos_frame["target_joint_min"]' in daily_source
        and 'oos_frame["joint_min_score"]' in daily_source,
    )
    check_true(
        "mr13w_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_joint_attention_pool_contract_case(_base_params):
    """Protect MR-13X as the MR-13W target + joint-only temporal-pooling contrast."""

    case_id = "BREAKOUT_QUALITY_JOINT_ATTENTION_POOL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13x_historical_mr13w_joint_pooling_only_contrast_remains_registered",
        (
            "MR-13X",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
            "inception_time_safety_raw_mfe_joint_attn_mlp_v1",
            "raw_mfe_mean_daily_spearman",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            profile.epoch_selection_metric,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check_true(
        "mr13x_keeps_mr13w_target_loss_epoch_and_pairwise_reduction",
        profile.training_objective == control.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )

    control_spec = get_model_spec(control.model_architecture)
    attention_spec = get_model_spec(profile.model_architecture)
    trunk_fields = (
        "inception_depth",
        "inception_filters",
        "inception_bottleneck_channels",
        "inception_kernel_sizes",
        "inception_residual_every",
        "normalization",
        "normalization_groups",
        "dropout",
        "head_width",
        "use_dataset_context",
        "sequence_input_paths",
    )
    check_true(
        "mr13x_keeps_raw_300x10_mr13w_trunk_and_joint_mlp_spec",
        len(FEATURE_COLUMNS) == 10
        and all(getattr(control_spec, field) == getattr(attention_spec, field) for field in trunk_fields)
        and tuple(control_spec.pooling)
        == (
            "global_average",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "direct_hmhs_mlp_head",
        )
        and tuple(attention_spec.pooling)
        == (
            "global_average_for_marginal_heads",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "joint_scalar_attention_pool",
            "joint_mlp_head",
        ),
    )

    torch, nn = require_torch()
    torch.manual_seed(42)
    control_model = build_active_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=control.model_architecture,
    )
    torch.manual_seed(42)
    attention_model = build_active_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=profile.model_architecture,
    )
    control_state = control_model.state_dict()
    attention_state = attention_model.state_dict()
    shared_keys = sorted(set(control_state) & set(attention_state))
    extra_keys = sorted(set(attention_state) - set(control_state))
    check_true(
        "mr13x_same_seed_changes_only_attention_scorer_parameters_at_initialization",
        not (set(control_state) - set(attention_state))
        and extra_keys
        == ["joint_attention_scorer.bias", "joint_attention_scorer.weight"]
        and all(torch.equal(control_state[key], attention_state[key]) for key in shared_keys),
    )
    scorer = attention_model.joint_attention_scorer
    check_true(
        "mr13x_joint_pool_is_single_scalar_1x1_attention_without_extra_hyperparameters",
        isinstance(scorer, nn.Conv1d)
        and int(scorer.in_channels) == int(attention_spec.head_width)
        and int(scorer.out_channels) == 1
        and tuple(scorer.kernel_size) == (1,)
        and tuple(scorer.stride) == (1,)
        and tuple(scorer.padding) == (0,)
        and isinstance(attention_model.joint_hmhs_classifier, nn.Sequential)
        and len(attention_model.joint_hmhs_classifier) == 3,
    )

    torch.manual_seed(7)
    x = torch.randn(4, 64, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    control_model.eval()
    attention_model.eval()
    with torch.no_grad():
        weights = attention_model.joint_attention_weights(x)
        control_heads = control_model.forward_safety_raw_mfe_hmhs_heads(x, context)
        attention_heads = attention_model.forward_safety_raw_mfe_hmhs_heads(x, context)
    check_true(
        "mr13x_attention_weights_are_per_bar_softmax_and_marginal_logits_are_identical",
        tuple(weights.shape) == (4, 64)
        and torch.all(weights >= 0)
        and torch.allclose(weights.float().sum(dim=1), torch.ones(4), atol=1e-6)
        and torch.equal(control_heads[0], attention_heads[0])
        and torch.equal(control_heads[1], attention_heads[1])
        and not torch.equal(control_heads[2], attention_heads[2]),
    )

    attention_model.zero_grad(set_to_none=True)
    joint_logits = attention_model.forward_safety_raw_mfe_hmhs_heads(x, context)[2]
    joint_logits.sum().backward()
    shared_parameter = next(attention_model.inception_modules[0].parameters())
    check_true(
        "mr13x_joint_loss_updates_attention_joint_mlp_and_shared_trunk_not_marginal_classifiers",
        scorer.weight.grad is not None
        and scorer.bias.grad is not None
        and all(parameter.grad is not None for parameter in attention_model.joint_hmhs_classifier.parameters())
        and shared_parameter.grad is not None
        and attention_model.raw_safety_classifier.weight.grad is None
        and attention_model.conditional_mfe_classifier.weight.grad is None,
    )

    project_root = Path(__file__).resolve().parents[2]
    check_true(
        "mr13x_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_modern_tcn_joint_min_contract_case(_base_params):
    """Protect MR-13Y as the Joint-Min raw-data trunk-family comparison."""

    case_id = "BREAKOUT_QUALITY_MODERN_TCN_JOINT_MIN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.factory import build_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import (
        ACTIVE_MODEL_ARCHITECTURES,
        MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
        MODERN_TCN_V1,
        get_model_spec,
    )

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_MODERN_TCN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13y_historical_joint_min_modern_tcn_architecture_family_contrast_remains_registered",
        (
            "MR-13Y",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
            MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
            "raw_mfe_mean_daily_spearman",
            False,
            False,
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            profile.epoch_selection_metric,
            spec.selection_pit_authorized,
            spec.current_time_validation_authorized,
        ),
    )
    check_true(
        "mr13y_keeps_mr13x_target_loss_epoch_and_pairwise_reduction",
        profile.training_objective == control.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )

    modern_spec = get_model_spec(profile.model_architecture)
    legacy_spec = get_model_spec(MODERN_TCN_V1)
    check_true(
        "mr13y_uses_new_active_architecture_identity_while_historical_9b_stays_legacy",
        MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 in ACTIVE_MODEL_ARCHITECTURES
        and MODERN_TCN_V1 not in ACTIVE_MODEL_ARCHITECTURES
        and modern_spec.architecture != legacy_spec.architecture
        and modern_spec.family == "modern_tcn_safety_raw_mfe_joint_attn_mlp",
    )
    check(
        "mr13y_reuses_historical_9b_modern_tcn_trunk_recipe_without_hyperparameter_sweep",
        (6, 96, 51, 4, "batch_norm", 0.10, 301, ("raw_level",), False),
        (
            modern_spec.modern_tcn_depth,
            modern_spec.modern_tcn_channels,
            modern_spec.modern_tcn_kernel_size,
            modern_spec.modern_tcn_expansion_ratio,
            modern_spec.normalization,
            modern_spec.dropout,
            modern_spec.receptive_field_bars,
            tuple(modern_spec.sequence_input_paths),
            bool(modern_spec.use_dataset_context),
        ),
    )
    check_true(
        "mr13y_preserves_joint_min_head_semantics_with_latent_width_coupled_capacity",
        len(FEATURE_COLUMNS) == 10
        and int(modern_spec.head_width or 0) == 96
        and tuple(modern_spec.pooling)
        == (
            "global_average_for_marginal_heads",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "joint_scalar_attention_pool",
            "joint_mlp_head",
        ),
    )

    torch, nn = require_torch()
    torch.manual_seed(42)
    legacy_model = build_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=MODERN_TCN_V1,
    )
    torch.manual_seed(42)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=profile.model_architecture,
    )
    legacy_state = legacy_model.state_dict()
    state = model.state_dict()
    trunk_keys = sorted(
        key for key in legacy_state if key.startswith("stem.") or key.startswith("blocks.")
    )
    check_true(
        "mr13y_same_seed_reconstructs_historical_9b_trunk_weights_exactly",
        bool(trunk_keys)
        and all(key in state and torch.equal(legacy_state[key], state[key]) for key in trunk_keys),
    )
    check_true(
        "mr13y_head_and_attention_topology_matches_mr13x_semantics_at_96d_latent",
        isinstance(model.raw_safety_classifier, nn.Linear)
        and int(model.raw_safety_classifier.in_features) == 96
        and int(model.raw_safety_classifier.out_features) == 2
        and isinstance(model.conditional_mfe_classifier, nn.Linear)
        and int(model.conditional_mfe_classifier.in_features) == 97
        and int(model.conditional_mfe_classifier.out_features) == 2
        and isinstance(model.joint_hmhs_classifier, nn.Sequential)
        and len(model.joint_hmhs_classifier) == 3
        and int(model.joint_hmhs_classifier[0].in_features) == 96
        and int(model.joint_hmhs_classifier[0].out_features) == 96
        and isinstance(model.joint_hmhs_classifier[1], nn.ReLU)
        and int(model.joint_hmhs_classifier[2].in_features) == 96
        and int(model.joint_hmhs_classifier[2].out_features) == 2
        and isinstance(model.joint_attention_scorer, nn.Conv1d)
        and int(model.joint_attention_scorer.in_channels) == 96
        and int(model.joint_attention_scorer.out_channels) == 1
        and tuple(model.joint_attention_scorer.kernel_size) == (1,),
    )

    torch.manual_seed(17)
    x = torch.randn(4, 64, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    model.eval()
    with torch.no_grad():
        weights = model.joint_attention_weights(x)
        heads = model.forward_safety_raw_mfe_hmhs_heads(x, context)
        tri_logits = model.forward_output_head(x, context, "tri_head")
    check_true(
        "mr13y_attention_softmax_and_tri_head_output_surface_match_production_contract",
        tuple(weights.shape) == (4, 64)
        and torch.all(weights >= 0)
        and torch.allclose(weights.float().sum(dim=1), torch.ones(4), atol=1e-6)
        and tuple(heads[0].shape) == (4, 2)
        and tuple(heads[1].shape) == (4, 2)
        and tuple(heads[2].shape) == (4, 2)
        and tuple(tri_logits.shape) == (4, 6),
    )

    model.zero_grad(set_to_none=True)
    joint_logits = model.forward_safety_raw_mfe_hmhs_heads(x, context)[2]
    joint_logits.sum().backward()
    shared_parameter = next(model.blocks[0].parameters())
    check_true(
        "mr13y_joint_loss_updates_attention_joint_mlp_and_modern_tcn_trunk_not_marginal_classifiers",
        model.joint_attention_scorer.weight.grad is not None
        and model.joint_attention_scorer.bias.grad is not None
        and all(parameter.grad is not None for parameter in model.joint_hmhs_classifier.parameters())
        and shared_parameter.grad is not None
        and model.raw_safety_classifier.weight.grad is None
        and model.conditional_mfe_classifier.weight.grad is None,
    )

    project_root = Path(__file__).resolve().parents[2]
    check_true(
        "mr13y_historical_model_remains_not_c75_source_after_mr13z_conversion",
        not _strategy_c75_uses_experiment_profile(profile.name),
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_patch_transformer_joint_min_contract_case(_base_params):
    """Protect MR-13Z as the frozen-recipe Patch-Transformer family comparison."""

    case_id = "BREAKOUT_QUALITY_PATCH_TRANSFORMER_JOINT_MIN"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.factory import build_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import (
        ACTIVE_MODEL_ARCHITECTURES,
        PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
        PATCH_TRANSFORMER_V1,
        get_model_spec,
        validate_model_sequence_length,
    )

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13z_joint_min_patch_transformer_architecture_family_contrast_remains_registered",
        (
            "MR-13Z",
            TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING,
            PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1,
            "raw_mfe_mean_daily_spearman",
        ),
        (
            spec.model_research_id,
            profile.training_objective,
            profile.model_architecture,
            profile.epoch_selection_metric,
        ),
    )
    check_true(
        "mr13z_keeps_mr13x_target_loss_epoch_and_pairwise_reduction",
        profile.training_objective == control.training_objective
        == TRAINING_OBJECTIVE_DAILY_SAFETY_RAW_MFE_JOINT_MIN_PAIRWISE_RANKING
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )

    patch_spec = get_model_spec(profile.model_architecture)
    legacy_spec = get_model_spec(PATCH_TRANSFORMER_V1)
    check_true(
        "mr13z_uses_new_active_architecture_identity_while_historical_9f_stays_legacy",
        PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1
        in ACTIVE_MODEL_ARCHITECTURES
        and PATCH_TRANSFORMER_V1 not in ACTIVE_MODEL_ARCHITECTURES
        and patch_spec.architecture != legacy_spec.architecture
        and patch_spec.family
        == "patch_token_transformer_safety_raw_mfe_joint_attn_mlp",
    )
    check(
        "mr13z_reuses_historical_9f_patch_transformer_recipe_without_hyperparameter_sweep",
        (10, 10, 128, 3, 4, 256, "mean", "sinusoidal", "layer_norm", 0.10, 300, False),
        (
            patch_spec.patch_transformer_patch_size,
            patch_spec.patch_transformer_patch_stride,
            patch_spec.patch_transformer_embedding_dim,
            patch_spec.patch_transformer_depth,
            patch_spec.patch_transformer_heads,
            patch_spec.patch_transformer_mlp_dim,
            patch_spec.patch_transformer_pooling,
            patch_spec.patch_transformer_positional_encoding,
            patch_spec.normalization,
            patch_spec.dropout,
            patch_spec.receptive_field_bars,
            bool(patch_spec.use_dataset_context),
        ),
    )
    check_true(
        "mr13z_preserves_joint_min_head_semantics_at_128d_patch_token_latent",
        len(FEATURE_COLUMNS) == 10
        and int(patch_spec.head_width or 0) == 128
        and tuple(patch_spec.pooling)
        == (
            "patch_token_global_average_for_marginal_heads",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "joint_scalar_attention_pool_over_patch_tokens",
            "joint_mlp_head",
        ),
    )
    validate_model_sequence_length(patch_spec, 300)
    invalid_sequence_rejected = False
    try:
        validate_model_sequence_length(patch_spec, 295)
    except ValueError:
        invalid_sequence_rejected = True
    check_true(
        "mr13z_patch_sequence_contract_requires_historical_nonoverlap_divisibility",
        invalid_sequence_rejected,
    )

    torch, nn = require_torch()
    torch.manual_seed(42)
    legacy_model = build_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=PATCH_TRANSFORMER_V1,
    )
    torch.manual_seed(42)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS), context_count=0,
        architecture=profile.model_architecture,
    )
    legacy_state = legacy_model.state_dict()
    state = model.state_dict()
    trunk_prefixes = (
        "patch_projection.",
        "patch_normalization.",
        "encoder.",
        "output_normalization.",
    )
    trunk_keys = sorted(key for key in legacy_state if key.startswith(trunk_prefixes))
    check_true(
        "mr13z_same_seed_reconstructs_historical_9f_patch_transformer_trunk_exactly",
        bool(trunk_keys)
        and all(key in state and torch.equal(legacy_state[key], state[key]) for key in trunk_keys),
    )
    check_true(
        "mr13z_head_and_attention_topology_matches_mr13x_semantics_at_128d_latent",
        isinstance(model.raw_safety_classifier, nn.Linear)
        and int(model.raw_safety_classifier.in_features) == 128
        and int(model.raw_safety_classifier.out_features) == 2
        and isinstance(model.conditional_mfe_classifier, nn.Linear)
        and int(model.conditional_mfe_classifier.in_features) == 129
        and int(model.conditional_mfe_classifier.out_features) == 2
        and isinstance(model.joint_hmhs_classifier, nn.Sequential)
        and len(model.joint_hmhs_classifier) == 3
        and int(model.joint_hmhs_classifier[0].in_features) == 128
        and int(model.joint_hmhs_classifier[0].out_features) == 128
        and isinstance(model.joint_hmhs_classifier[1], nn.ReLU)
        and int(model.joint_hmhs_classifier[2].in_features) == 128
        and int(model.joint_hmhs_classifier[2].out_features) == 2
        and isinstance(model.joint_attention_scorer, nn.Conv1d)
        and int(model.joint_attention_scorer.in_channels) == 128
        and int(model.joint_attention_scorer.out_channels) == 1
        and tuple(model.joint_attention_scorer.kernel_size) == (1,),
    )

    torch.manual_seed(19)
    x = torch.randn(4, 300, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    model.eval()
    with torch.no_grad():
        token_map = model.encode_token_map(x)
        weights = model.joint_attention_weights(x)
        heads = model.forward_safety_raw_mfe_hmhs_heads(x, context)
        tri_logits = model.forward_output_head(x, context, "tri_head")
    check_true(
        "mr13z_patch_token_attention_softmax_and_tri_head_output_surface_match_production_contract",
        tuple(token_map.shape) == (4, 30, 128)
        and tuple(weights.shape) == (4, 30)
        and torch.all(weights >= 0)
        and torch.allclose(weights.float().sum(dim=1), torch.ones(4), atol=1e-6)
        and tuple(heads[0].shape) == (4, 2)
        and tuple(heads[1].shape) == (4, 2)
        and tuple(heads[2].shape) == (4, 2)
        and tuple(tri_logits.shape) == (4, 6),
    )

    model.zero_grad(set_to_none=True)
    model.forward_safety_raw_mfe_hmhs_heads(x, context)[2].sum().backward()
    shared_parameter = model.patch_projection.weight
    check_true(
        "mr13z_joint_loss_updates_patch_trunk_attention_and_joint_mlp_not_marginal_classifiers",
        model.joint_attention_scorer.weight.grad is not None
        and model.joint_attention_scorer.bias.grad is not None
        and all(parameter.grad is not None for parameter in model.joint_hmhs_classifier.parameters())
        and shared_parameter.grad is not None
        and model.raw_safety_classifier.weight.grad is None
        and model.conditional_mfe_classifier.weight.grad is None,
    )

    from config.compatibility.strategy_compare_history import (
        HISTORICAL_STRATEGY_COMPARE_ARMS,
        HISTORICAL_STRATEGY_DL_SOURCES,
    )

    c75 = dict(HISTORICAL_STRATEGY_COMPARE_ARMS.get("C75") or {})
    cont13z = dict(HISTORICAL_STRATEGY_DL_SOURCES.get("CONT13Z_ROLL") or {})
    check_true(
        "mr13z_model_gate_pass_keeps_historical_c75_without_runtime_promotion",
        bool(spec.selection_pit_authorized)
        and bool(spec.current_time_validation_authorized)
        and c75.get("dl_id") == "CONT13Z_ROLL"
        and c75.get("dl_runtime_options", {}).get("primary_score_column") == "joint_min_score"
        and cont13z.get("experiment_profile")
        == DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_mr13aa_patch_transformer_h_target_contract_case(_base_params):
    """Protect MR-13AA as an architecture-only MR-13H target control."""

    case_id = "BREAKOUT_QUALITY_MR13AA_PATCH_TRANSFORMER_H_TARGET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.contract import FEATURE_COLUMNS
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.factory import build_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import (
        ACTIVE_MODEL_ARCHITECTURES,
        PATCH_TOKEN_TRANSFORMER_RANKER_V1,
        PATCH_TRANSFORMER_V1,
        get_model_spec,
        validate_model_sequence_length,
    )

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_NO_BREACH_PATCH_TRANSFORMER_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research_spec = get_continuous_ranker_research_spec(profile.name)
    patch_joint_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )

    check(
        "mr13aa_remains_registered_architecture_only_h_target_control",
        (
            "MR-13AA",
            TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
            "daily_full_horizon_opportunity_r_v1",
            PATCH_TOKEN_TRANSFORMER_RANKER_V1,
            "mean_daily_spearman",
        ),
        (
            research_spec.model_research_id,
            profile.training_objective,
            profile.continuous_target_id,
            profile.model_architecture,
            profile.epoch_selection_metric,
        ),
    )
    check_true(
        "mr13aa_keeps_exact_mr13h_training_target_universe_loss_and_selection_semantics",
        profile.optimizer_name == control.optimizer_name
        and profile.lr_schedule_name == control.lr_schedule_name
        and profile.augmentation_name == control.augmentation_name
        and profile.training_sampling_mode == control.training_sampling_mode
        and profile.training_objective == control.training_objective
        and profile.continuous_target_id == control.continuous_target_id
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and profile.training_label_scope == control.training_label_scope
        and profile.training_sample_scope == control.training_sample_scope
        and profile.model_architecture != control.model_architecture
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )
    check_true(
        "mr13aa_model_gate_only_has_no_pit_or_current_strategy_authorization",
        research_spec.reference_profile_name == control.name
        and not bool(research_spec.selection_pit_authorized)
        and not bool(research_spec.current_time_validation_authorized),
    )

    patch_spec = get_model_spec(profile.model_architecture)
    patch_joint_spec = get_model_spec(patch_joint_profile.model_architecture)
    legacy_spec = get_model_spec(PATCH_TRANSFORMER_V1)
    check_true(
        "mr13aa_uses_new_active_patch_ranker_identity_without_reviving_legacy_9f",
        PATCH_TOKEN_TRANSFORMER_RANKER_V1 in ACTIVE_MODEL_ARCHITECTURES
        and PATCH_TRANSFORMER_V1 not in ACTIVE_MODEL_ARCHITECTURES
        and patch_spec.architecture != legacy_spec.architecture
        and patch_spec.family == "patch_token_transformer_ranker",
    )
    check(
        "mr13aa_reuses_frozen_mr13z_patch_transformer_temporal_recipe_exactly",
        (
            patch_joint_spec.patch_transformer_patch_size,
            patch_joint_spec.patch_transformer_patch_stride,
            patch_joint_spec.patch_transformer_embedding_dim,
            patch_joint_spec.patch_transformer_depth,
            patch_joint_spec.patch_transformer_heads,
            patch_joint_spec.patch_transformer_mlp_dim,
            patch_joint_spec.patch_transformer_pooling,
            patch_joint_spec.patch_transformer_positional_encoding,
            patch_joint_spec.normalization,
            patch_joint_spec.dropout,
            patch_joint_spec.receptive_field_bars,
            bool(patch_joint_spec.use_dataset_context),
        ),
        (
            patch_spec.patch_transformer_patch_size,
            patch_spec.patch_transformer_patch_stride,
            patch_spec.patch_transformer_embedding_dim,
            patch_spec.patch_transformer_depth,
            patch_spec.patch_transformer_heads,
            patch_spec.patch_transformer_mlp_dim,
            patch_spec.patch_transformer_pooling,
            patch_spec.patch_transformer_positional_encoding,
            patch_spec.normalization,
            patch_spec.dropout,
            patch_spec.receptive_field_bars,
            bool(patch_spec.use_dataset_context),
        ),
    )
    check_true(
        "mr13aa_keeps_single_score_mean_pooling_surface",
        tuple(patch_spec.pooling) == ("patch_token_global_average", "single_rank_head")
        and len(FEATURE_COLUMNS) == 10,
    )
    validate_model_sequence_length(patch_spec, 300)
    invalid_sequence_rejected = False
    try:
        validate_model_sequence_length(patch_spec, 295)
    except ValueError:
        invalid_sequence_rejected = True
    check_true(
        "mr13aa_patch_sequence_contract_is_300_bars_to_30_nonoverlap_tokens",
        invalid_sequence_rejected,
    )

    torch, nn = require_torch()
    torch.manual_seed(42)
    legacy_model = build_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=PATCH_TRANSFORMER_V1,
    )
    torch.manual_seed(42)
    joint_model = build_active_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=patch_joint_profile.model_architecture,
    )
    torch.manual_seed(42)
    model = build_active_model(
        feature_count=len(FEATURE_COLUMNS),
        context_count=0,
        architecture=profile.model_architecture,
    )
    trunk_prefixes = (
        "patch_projection.",
        "patch_normalization.",
        "encoder.",
        "output_normalization.",
    )
    legacy_state = legacy_model.state_dict()
    joint_state = joint_model.state_dict()
    state = model.state_dict()
    trunk_keys = sorted(key for key in legacy_state if key.startswith(trunk_prefixes))
    check_true(
        "mr13aa_same_seed_trunk_exactly_matches_legacy_9f_and_mr13z_frozen_trunk",
        bool(trunk_keys)
        and all(
            key in state
            and key in joint_state
            and torch.equal(legacy_state[key], state[key])
            and torch.equal(joint_state[key], state[key])
            for key in trunk_keys
        ),
    )
    check_true(
        "mr13aa_has_only_single_linear_rank_head_after_patch_encoder",
        isinstance(model.dropout, nn.Dropout)
        and abs(float(model.dropout.p) - 0.10) < 1e-12
        and isinstance(model.classifier, nn.Linear)
        and int(model.classifier.in_features) == 128
        and int(model.classifier.out_features) == 2
        and not hasattr(model, "raw_safety_classifier")
        and not hasattr(model, "conditional_mfe_classifier")
        and not hasattr(model, "joint_hmhs_classifier")
        and not hasattr(model, "joint_attention_scorer"),
    )

    torch.manual_seed(23)
    x = torch.randn(4, 300, len(FEATURE_COLUMNS))
    context = torch.empty(4, 0)
    model.eval()
    with torch.no_grad():
        token_map = model.encode_token_map(x)
        logits = model(x, context)
    check_true(
        "mr13aa_forward_surface_is_30x128_patch_tokens_to_two_logits",
        tuple(token_map.shape) == (4, 30, 128)
        and tuple(logits.shape) == (4, 2),
    )


    summary["training_performed"] = False
    return results, summary



def validate_breakout_quality_mr13ab_first_breach_pure_mfe_contract_case(_base_params):
    """Protect MR-13AB as the missing Pure-MFE × first-risk-breach target cell."""

    case_id = "BREAKOUT_QUALITY_MR13AB_FIRST_BREACH_PURE_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd

    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE,
        DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.continuous_target import (
        DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
        StrategyAlignedContinuousTargetSpec,
        build_daily_first_risk_breach_pure_mfe_contract,
        daily_first_risk_breach_pure_mfe_target_from_cached_path,
        daily_full_horizon_pure_mfe_target_from_cached_path,
    )
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
    from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
    from services.breakout_quality.daily_target_comparison import _controlled_change_contract

    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research_spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13ab_historical_identity_remains_exact_after_current_profile_moves_on",
        (
            profile.name,
            "MR-13AB",
            TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
            DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
            "inception_time_v1",
            "mean_daily_spearman",
        ),
        (
            research_spec.profile_name,
            research_spec.model_research_id,
            profile.training_objective,
            profile.continuous_target_id,
            profile.model_architecture or "inception_time_v1",
            profile.epoch_selection_metric,
        ),
    )
    check_true(
        "mr13ab_keeps_mr13k_training_universe_loss_and_recipe_exact",
        profile.optimizer_name == control.optimizer_name
        and profile.lr_schedule_name == control.lr_schedule_name
        and profile.augmentation_name == control.augmentation_name
        and profile.training_sampling_mode == control.training_sampling_mode
        and profile.training_objective == control.training_objective
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and profile.training_label_scope == control.training_label_scope
        and profile.training_sample_scope == control.training_sample_scope
        and profile.model_architecture == control.model_architecture
        and profile.continuous_target_id != control.continuous_target_id
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )
    check_true(
        "mr13ab_model_gate_only_has_no_pit_or_current_strategy_authorization",
        research_spec.reference_profile_name == control.name
        and not bool(research_spec.selection_pit_authorized)
        and not bool(research_spec.current_time_validation_authorized),
    )

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    horizon = int(spec.horizon_bars)
    no_breach_high = np.asarray([101.0, 102.0, 103.0, 104.0] + [104.0] * (horizon - 4))
    no_breach_low = np.asarray([99.0] * horizon)
    candidate_no_breach = daily_first_risk_breach_pure_mfe_target_from_cached_path(
        no_breach_high, no_breach_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    control_no_breach = daily_full_horizon_pure_mfe_target_from_cached_path(
        no_breach_high, no_breach_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "mr13ab_no_breach_path_is_exact_mr13k_pure_mfe",
        (
            control_no_breach.valid,
            control_no_breach.target_raw_r,
            control_no_breach.favorable_return,
            control_no_breach.adverse_return_to_peak,
            control_no_breach.opportunity_bar,
            control_no_breach.first_risk_breach_bar,
        ),
        (
            candidate_no_breach.valid,
            candidate_no_breach.target_raw_r,
            candidate_no_breach.favorable_return,
            candidate_no_breach.adverse_return_to_peak,
            candidate_no_breach.opportunity_bar,
            candidate_no_breach.first_risk_breach_bar,
        ),
    )

    breach_high = np.asarray([102.0, 103.0, 150.0] + [160.0] * (horizon - 3))
    breach_low = np.asarray([99.0, 98.0, 89.0] + [88.0] * (horizon - 3))
    breached = daily_first_risk_breach_pure_mfe_target_from_cached_path(
        breach_high, breach_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "mr13ab_same_bar_adverse_first_excludes_breach_day_high_and_post_breach_upside",
        (True, 3, 2, 0.03, 0.30, 0.02),
        (
            bool(breached.valid),
            int(breached.first_risk_breach_bar),
            int(breached.opportunity_bar),
            round(float(breached.favorable_return), 8),
            round(float(breached.target_raw_r), 8),
            round(float(breached.adverse_return_to_peak), 8),
        ),
    )

    immediate_high = np.asarray([150.0] + [160.0] * (horizon - 1))
    immediate_low = np.asarray([89.0] + [88.0] * (horizon - 1))
    immediate = daily_first_risk_breach_pure_mfe_target_from_cached_path(
        immediate_high, immediate_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check(
        "mr13ab_first_bar_breach_has_zero_usable_mfe_not_minus_one_r",
        (True, 1, 1, 0.0, 0.0, round(float(spec.risk_budget_return), 8)),
        (
            bool(immediate.valid),
            int(immediate.first_risk_breach_bar),
            int(immediate.opportunity_bar),
            round(float(immediate.favorable_return), 8),
            round(float(immediate.target_raw_r), 8),
            round(float(immediate.adverse_return_to_peak), 8),
        ),
    )

    negative_high = np.asarray([99.0] * horizon)
    negative_low = np.asarray([89.0] + [88.0] * (horizon - 1))
    immediate_zero_floor = daily_first_risk_breach_pure_mfe_target_from_cached_path(
        negative_high, negative_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    historical_unfloored = daily_full_horizon_pure_mfe_target_from_cached_path(
        negative_high, negative_low,
        anchor_price=100.0, available_bars=horizon, spec=spec,
    )
    check_true(
        "mr13ab_first_bar_empty_prebreach_zero_can_exceed_historical_unfloored_negative_mr13k_pure_mfe",
        bool(
            immediate_zero_floor.valid
            and historical_unfloored.valid
            and float(immediate_zero_floor.target_raw_r) == 0.0
            and float(historical_unfloored.target_raw_r) < 0.0
            and float(immediate_zero_floor.target_raw_r) > float(historical_unfloored.target_raw_r)
        ),
    )

    frame = pd.DataFrame(
        {
            "Close": np.asarray([100.0] + [100.0] * horizon),
            "High": np.asarray([100.0] + breach_high.tolist()),
            "Low": np.asarray([100.0] + breach_low.tolist()),
        }
    )
    batch = compute_daily_opportunity_target_batch(
        frame,
        np.asarray([0], dtype=np.int64),
        spec=spec,
        target_id=DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
    )
    check(
        "mr13ab_vectorized_daily_builder_matches_scalar_first_passage_semantics",
        (
            np.float32(breached.target_raw_r),
            np.float32(breached.favorable_return),
            np.float32(breached.adverse_return_to_peak),
            np.int16(breached.opportunity_bar),
            np.int16(breached.first_risk_breach_bar),
        ),
        (
            batch.target_raw_r[0],
            batch.favorable_return[0],
            batch.adverse_return_to_peak[0],
            batch.opportunity_bar[0],
            batch.first_risk_breach_bar[0],
        ),
    )

    contract = build_daily_first_risk_breach_pure_mfe_contract(DEFAULT_LABEL_POLICY)
    check_true(
        "mr13ab_target_contract_discloses_first_passage_without_adverse_magnitude_penalty",
        contract["target_id"] == DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID
        and contract["risk_rule"] == "same-bar adverse-first; barrier-day high is excluded"
        and contract["adverse_penalty_included"] is False
        and contract["no_safe_bar_rule"] == "target=0R when the risk barrier is touched on the first bar"
        and "favorable_return_before_first_risk_breach" in str(contract["formula"]),
    )
    change = _controlled_change_contract(
        DAILY_FIRST_RISK_BREACH_PURE_MFE_TARGET_ID,
        str(control.continuous_target_id),
    )
    check_true(
        "mr13ab_reference_comparison_is_target_only_against_mr13k",
        change["change_id"] == "add_first_risk_breach_path_truncation_to_pure_mfe_only"
        and bool(change["enforce_no_breach_invariant"])
        and not bool(change["enforce_same_peak_components"]),
    )

    project_root = Path(__file__).resolve().parents[2]
    strategy_source = (project_root / "config" / "strategy_compare.py").read_text(encoding="utf-8")
    check_true(
        "mr13ab_has_no_strategy_conversion_or_runtime_source",
        "daily_universal_first_risk_breach_pure_mfe_full_list_ndcg_pairwise" not in strategy_source
        and "MR-13AB" not in strategy_source,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_mr13ac_predicted_upside_conditional_safety_contract_case(_base_params):
    """Protect MR-13AC PIT-safe predicted-upside conditional low-adverse semantics."""

    case_id = "BREAKOUT_QUALITY_MR13AC_PREDICTED_UPSIDE_CONDITIONAL_SAFETY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd

    from config.breakout_quality import (
        DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.conditional_mfe_safety import (
        build_conditional_mfe_safety_targets,
        build_same_date_residual_percentile,
    )
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.predicted_upside_context import (
        PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
        STAGE1_ARCHITECTURE,
        STAGE1_PROFILE,
        STAGE1_RESEARCH_ID,
        STAGE1_SEED,
        build_predicted_upside_conditional_low_adverse_targets,
        predicted_upside_context_contract,
    )
    from filters.breakout_quality.ranker_training_contract import training_semantics

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research_spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13ac_registered_identity_target_and_architecture",
        (
            "MR-13AC",
            PREDICTED_UPSIDE_CONDITIONAL_LOW_ADVERSE_TARGET_ID,
            "inception_time_predicted_upside_context_v1",
        ),
        (
            research_spec.model_research_id,
            profile.continuous_target_id,
            profile.model_architecture,
        ),
    )
    check_true(
        "mr13ac_keeps_mr13m_stage2_training_recipe_except_target_and_one_context_scalar",
        profile.optimizer_name == control.optimizer_name
        and profile.lr_schedule_name == control.lr_schedule_name
        and profile.augmentation_name == control.augmentation_name
        and profile.training_sampling_mode == control.training_sampling_mode
        and profile.training_objective == control.training_objective
        and profile.loss_name == control.loss_name
        and profile.epoch_selection_metric == control.epoch_selection_metric
        and profile.training_label_scope == control.training_label_scope
        and profile.training_sample_scope == control.training_sample_scope
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(control.name).pairwise_reduction,
    )

    context_contract = predicted_upside_context_contract()
    check(
        "mr13ac_stage1_is_frozen_mr13k_seed42_with_crossfit_selection_and_fixed_forward",
        (
            "MR-13K",
            "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
            "inception_time_v1",
            42,
            "expanding_cross_fitted_point_in_time",
            "single_fixed_pre_oos_fit",
            True,
            True,
        ),
        (
            STAGE1_RESEARCH_ID,
            STAGE1_PROFILE,
            STAGE1_ARCHITECTURE,
            int(STAGE1_SEED),
            context_contract.get("selection_context"),
            context_contract.get("forward_context"),
            bool(context_contract.get("full_fit_selection_score_forbidden")),
            bool(context_contract.get("oos_statistics_for_training_forbidden")),
        ),
    )
    dates = pd.to_datetime(["2026-01-02"] * 6 + ["2026-01-05"] * 6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "target_adverse_r": [0.9, 0.2, 0.7, 0.1, 0.5, 0.4, 0.8, 0.3, 0.6, 0.2, 0.9, 0.1],
        }
    )
    context = np.asarray(
        [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        dtype=np.float32,
    )
    valid = np.ones(len(frame), dtype=bool)
    targets = build_predicted_upside_conditional_low_adverse_targets(frame, valid, context)
    orthogonal = True
    for _date, day in frame.groupby("date", sort=True):
        idx = day.index.to_numpy(dtype=np.int64)
        x = context[idx].astype(np.float64)
        residual = targets.residual[idx].astype(np.float64)
        orthogonal &= abs(float(np.mean(residual))) < 1e-6
        orthogonal &= abs(float(np.dot(x - np.mean(x), residual))) < 1e-6
    check_true(
        "mr13ac_target_is_same_date_low_adverse_residual_orthogonal_to_predicted_upside_context",
        orthogonal
        and np.isfinite(targets.residual_percentile).all()
        and bool((targets.residual_percentile >= 0.0).all())
        and bool((targets.residual_percentile <= 1.0).all()),
    )
    check_true(
        "mr13ac_training_target_is_same_date_residual_percentile_not_raw_residual",
        np.array_equal(targets.training_target, targets.residual_percentile)
        and bool((targets.training_target >= 0.0).all())
        and bool((targets.training_target <= 1.0).all()),
    )
    check(
        "mr13ac_selection_stage2_universe_is_only_pit_context_covered_rows",
        "pit_context_covered_rows_only",
        context_contract.get("selection_training_universe"),
    )

    # Refactoring MR-13P onto the shared residual helper must preserve its exact transform.
    legacy_frame = frame.copy()
    legacy_frame["target_favorable_r"] = [1, 2, 3, 4, 5, 6] * 2
    mr13p = build_conditional_mfe_safety_targets(legacy_frame, valid)
    manual_residual, manual_percentile = build_same_date_residual_percentile(
        mr13p.primary_mfe_percentile,
        mr13p.low_adverse_percentile,
        valid,
        legacy_frame["date"],
    )
    check_true(
        "mr13ac_shared_residual_helper_keeps_mr13p_transform_exact",
        np.array_equal(mr13p.conditional_safety_residual, manual_residual)
        and np.array_equal(mr13p.conditional_safety_percentile, manual_percentile),
    )

    base_spec = get_model_spec("inception_time_v1")
    model_spec = get_model_spec("inception_time_predicted_upside_context_v1")
    check_true(
        "mr13ac_backbone_is_mr13m_inceptiontime_plus_exactly_one_direct_scalar_context",
        model_spec.inception_depth == base_spec.inception_depth
        and model_spec.inception_filters == base_spec.inception_filters
        and model_spec.inception_bottleneck_channels == base_spec.inception_bottleneck_channels
        and model_spec.inception_kernel_sizes == base_spec.inception_kernel_sizes
        and model_spec.inception_residual_every == base_spec.inception_residual_every
        and model_spec.dropout == base_spec.dropout
        and bool(model_spec.use_dataset_context)
        and model_spec.pooling == ("global_average", "predicted_upside_percentile_concat")
        and model_spec.head_width is None,
    )
    torch, _nn = require_torch()
    torch.manual_seed(29)
    model = build_active_model(
        feature_count=10,
        context_count=1,
        architecture="inception_time_predicted_upside_context_v1",
    )
    model.eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.weight[1, -1] = 1.0
        x = torch.randn(1, 300, 10).repeat(2, 1, 1)
        logits = model(x, torch.tensor([[0.2], [0.8]], dtype=x.dtype))
    check_true(
        "mr13ac_model_has_one_two_logit_head_and_context_enters_only_as_direct_scalar_concat",
        bool(getattr(model, "direct_context_concat", False))
        and int(model.classifier.out_features) == 2
        and int(model.classifier.in_features) == int(base_spec.inception_filters) * 4 + 1
        and abs(float(logits[1, 1] - logits[0, 1]) - 0.6) < 1e-5
        and not hasattr(model, "predicted_upside_context_network"),
    )
    rejected_bad_width = False
    try:
        build_active_model(
            feature_count=10,
            context_count=0,
            architecture="inception_time_predicted_upside_context_v1",
        )
    except ValueError:
        rejected_bad_width = True
    check_true("mr13ac_model_rejects_missing_context_scalar", rejected_bad_width)

    semantics = training_semantics(profile)
    embedded = dict(semantics.get("predicted_upside_context_contract") or {})
    if not embedded:
        embedded = dict(
            (semantics.get("pairwise_contract") or {}).get("predicted_upside_context_contract") or {}
        )
    check_true(
        "mr13ac_training_semantics_persist_stage1_context_provenance",
        embedded == context_contract,
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_mr13ad_predicted_safety_conditional_mfe_contract_case(_base_params):
    """Protect MR-13AD PIT-safe predicted-safety conditional MFE reverse-control semantics."""

    case_id = "BREAKOUT_QUALITY_MR13AD_PREDICTED_SAFETY_CONDITIONAL_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd

    from config.breakout_quality import (
        DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )
    from filters.breakout_quality.continuous_ranker_data import (
        build_same_date_percentile_targets,
    )
    from filters.breakout_quality.models.active import build_active_model
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.predicted_safety_context import (
        PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
        STAGE1_ARCHITECTURE,
        STAGE1_PROFILE,
        STAGE1_RESEARCH_ID,
        STAGE1_SEED,
        build_predicted_safety_conditional_mfe_targets,
        predicted_safety_context_contract,
    )
    from filters.breakout_quality.ranker_training_contract import training_semantics

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    mfe_control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    safety_control = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research_spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13ad_registered_identity_target_and_architecture",
        (
            profile.name,
            "MR-13AD",
            PREDICTED_SAFETY_CONDITIONAL_MFE_TARGET_ID,
            "inception_time_predicted_safety_context_v1",
        ),
        (
            profile.name,
            research_spec.model_research_id,
            profile.continuous_target_id,
            profile.model_architecture,
        ),
    )
    check_true(
        "mr13ad_keeps_mr13k_stage2_training_recipe_except_target_and_one_context_scalar",
        profile.optimizer_name == mfe_control.optimizer_name
        and profile.lr_schedule_name == mfe_control.lr_schedule_name
        and profile.augmentation_name == mfe_control.augmentation_name
        and profile.training_sampling_mode == mfe_control.training_sampling_mode
        and profile.training_objective == mfe_control.training_objective
        and profile.loss_name == mfe_control.loss_name
        and profile.epoch_selection_metric == mfe_control.epoch_selection_metric
        and profile.training_label_scope == mfe_control.training_label_scope
        and profile.training_sample_scope == mfe_control.training_sample_scope
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(mfe_control.name).pairwise_reduction,
    )

    context_contract = predicted_safety_context_contract()
    check(
        "mr13ad_stage1_is_frozen_mr13m_seed42_with_crossfit_selection_and_fixed_forward",
        (
            "MR-13M",
            DAILY_UNIVERSAL_FULL_HORIZON_LOW_ADVERSE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "inception_time_v1",
            42,
            "expanding_cross_fitted_point_in_time",
            "single_fixed_pre_oos_fit",
            True,
            True,
        ),
        (
            STAGE1_RESEARCH_ID,
            STAGE1_PROFILE,
            STAGE1_ARCHITECTURE,
            int(STAGE1_SEED),
            context_contract.get("selection_context"),
            context_contract.get("forward_context"),
            bool(context_contract.get("full_fit_selection_score_forbidden")),
            bool(context_contract.get("oos_statistics_for_training_forbidden")),
        ),
    )
    check_true(
        "mr13ad_stage1_profile_is_exact_mr13m_low_adverse_recipe",
        safety_control.name == STAGE1_PROFILE
        and get_continuous_ranker_research_spec(safety_control.name).model_research_id == "MR-13M",
    )
    dates = pd.to_datetime(["2026-01-02"] * 6 + ["2026-01-05"] * 6)
    frame = pd.DataFrame(
        {
            "date": dates,
            "target_favorable_r": [0.2, 0.8, 1.4, 2.6, 4.0, 7.5, 0.1, 0.6, 1.1, 2.1, 3.8, 9.0],
        }
    )
    context = np.asarray(
        [0.9, 0.2, 0.7, 0.4, 0.6, 0.1, 0.8, 0.3, 0.6, 0.2, 0.9, 0.4],
        dtype=np.float32,
    )
    valid = np.ones(len(frame), dtype=bool)
    targets = build_predicted_safety_conditional_mfe_targets(frame, valid, context)
    orthogonal = True
    for _date, day in frame.groupby("date", sort=True):
        idx = day.index.to_numpy(dtype=np.int64)
        x = context[idx].astype(np.float64)
        residual = targets.residual[idx].astype(np.float64)
        orthogonal &= abs(float(np.mean(residual))) < 1e-6
        orthogonal &= abs(float(np.dot(x - np.mean(x), residual))) < 1e-6
    manual_mfe = build_same_date_percentile_targets(
        frame["target_favorable_r"].to_numpy(dtype=np.float64), valid, frame["date"]
    )
    check_true(
        "mr13ad_target_is_same_date_pure_mfe_residual_orthogonal_to_predicted_safety_context",
        orthogonal
        and np.allclose(targets.pure_mfe_percentile, manual_mfe, atol=0.0, rtol=0.0)
        and np.isfinite(targets.residual_percentile).all()
        and bool((targets.residual_percentile >= 0.0).all())
        and bool((targets.residual_percentile <= 1.0).all()),
    )
    check_true(
        "mr13ad_training_target_is_same_date_residual_percentile_not_raw_residual",
        np.array_equal(targets.training_target, targets.residual_percentile)
        and bool((targets.training_target >= 0.0).all())
        and bool((targets.training_target <= 1.0).all()),
    )
    check_true(
        "mr13ad_reverse_control_uses_predicted_safety_context_not_true_future_safety_label",
        "target_adverse_r" not in frame.columns
        and context_contract.get("context_semantic")
        == "same_date_average_rank_percentile_of_stage1_predicted_low_adverse_safety"
        and context_contract.get("stage2_response") == "same_date_pure_mfe_percentile",
    )
    check(
        "mr13ad_selection_stage2_universe_is_only_pit_context_covered_rows",
        "pit_context_covered_rows_only",
        context_contract.get("selection_training_universe"),
    )

    base_spec = get_model_spec("inception_time_v1")
    model_spec = get_model_spec("inception_time_predicted_safety_context_v1")
    check_true(
        "mr13ad_backbone_is_mr13k_inceptiontime_plus_exactly_one_direct_scalar_context",
        model_spec.inception_depth == base_spec.inception_depth
        and model_spec.inception_filters == base_spec.inception_filters
        and model_spec.inception_bottleneck_channels == base_spec.inception_bottleneck_channels
        and model_spec.inception_kernel_sizes == base_spec.inception_kernel_sizes
        and model_spec.inception_residual_every == base_spec.inception_residual_every
        and model_spec.dropout == base_spec.dropout
        and bool(model_spec.use_dataset_context)
        and model_spec.pooling == ("global_average", "predicted_safety_percentile_concat")
        and model_spec.head_width is None,
    )
    torch, _nn = require_torch()
    torch.manual_seed(31)
    model = build_active_model(
        feature_count=10,
        context_count=1,
        architecture="inception_time_predicted_safety_context_v1",
    )
    model.eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        model.classifier.weight[1, -1] = 1.0
        x = torch.randn(1, 300, 10).repeat(2, 1, 1)
        logits = model(x, torch.tensor([[0.2], [0.8]], dtype=x.dtype))
    check_true(
        "mr13ad_model_has_one_two_logit_head_and_context_enters_only_as_direct_scalar_concat",
        bool(getattr(model, "direct_context_concat", False))
        and int(model.classifier.out_features) == 2
        and int(model.classifier.in_features) == int(base_spec.inception_filters) * 4 + 1
        and abs(float(logits[1, 1] - logits[0, 1]) - 0.6) < 1e-5
        and not hasattr(model, "predicted_safety_context_network"),
    )
    rejected_bad_width = False
    try:
        build_active_model(
            feature_count=10,
            context_count=0,
            architecture="inception_time_predicted_safety_context_v1",
        )
    except ValueError:
        rejected_bad_width = True
    check_true("mr13ad_model_rejects_missing_context_scalar", rejected_bad_width)

    semantics = training_semantics(profile)
    embedded = dict(semantics.get("predicted_safety_context_contract") or {})
    if not embedded:
        embedded = dict(
            (semantics.get("pairwise_contract") or {}).get("predicted_safety_context_contract") or {}
        )
    check_true(
        "mr13ad_training_semantics_persist_stage1_context_provenance",
        embedded == context_contract,
    )

    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_multi_dl_ranker_architecture_contract_case(_base_params):
    """Pin the profile-driven continuous-ranker boundary before adding new Daily MR variants."""

    case_id = "BREAKOUT_QUALITY_MULTI_DL_RANKER_ARCHITECTURE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_TARGET_GAP_WEIGHTED,
        CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
        CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE,
        DAILY_UNIVERSAL_NO_TIME_PERCENTILE_MSE_PROFILE,
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
    )

    continuous_profiles = tuple(
        name
        for name in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
        if get_breakout_quality_experiment_profile(name).training_objective
        in CONTINUOUS_RANKER_TRAINING_OBJECTIVES
    )
    check(
        "continuous_profile_and_research_spec_registry_are_one_to_one",
        continuous_profiles,
        tuple(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES),
    )

    mr12b = get_continuous_ranker_research_spec(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    mr13a = get_continuous_ranker_research_spec(DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE)
    mr13b = get_continuous_ranker_research_spec(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_GAP_WEIGHTED_PROFILE
    )
    check(
        "existing_pairwise_research_identity_and_reduction_remain_explicit",
        (
                    "MR-12B", CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
                    "MR-13A", CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL,
                    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
                    "daily_opportunity_rank",
                ),
        (
                    mr12b.model_research_id, mr12b.pairwise_reduction,
                    mr13a.model_research_id, mr13a.trainer_family,
                    mr13a.pairwise_reduction, mr13a.score_semantic_id,
                ),
    )

    event_recipe = get_continuous_ranker_execution_recipe(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    daily_recipe = get_continuous_ranker_execution_recipe(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    check(
        "daily_trainer_and_dispatch_are_execution_recipe_driven",
        ("event", CONTINUOUS_RANKER_TRAINER_DAILY_UNIVERSAL, False, False),
        (
            event_recipe.trainer_family,
            daily_recipe.trainer_family,
            hasattr(event_recipe, "model_research_id"),
            hasattr(daily_recipe, "phase"),
        ),
    )
    from config.breakout_quality import (
        TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
        TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
    )

    pairwise_specs = [
        get_continuous_ranker_research_spec(name)
        for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if get_breakout_quality_experiment_profile(name).training_objective
        in {
            TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_PARETO_PAIRWISE_RANKING,
            TRAINING_OBJECTIVE_DAILY_CONDITIONAL_MFE_SAFETY_PAIRWISE_RANKING,
        }
    ]
    non_default_owners = {}
    for spec in pairwise_specs:
        reduction = str(spec.pairwise_reduction)
        if reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR:
            continue
        non_default_owners.setdefault(reduction, []).append(spec.model_research_id)
    check_true(
        "non_default_pairwise_reductions_have_explicit_mr_owners_without_identity_collision",
        all(owners and len(owners) == len(set(owners)) for owners in non_default_owners.values()),
    )
    return results, summary




def validate_breakout_quality_mr13ae_predicted_safety_context_pure_mfe_contract_case(_base_params):
    """Protect MR-13AE: exact MR-13K Pure-MFE target plus PIT-safe predicted Safety context."""

    case_id = "BREAKOUT_QUALITY_MR13AE_PREDICTED_SAFETY_CONTEXT_PURE_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
        PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
        get_predicted_safety_pure_mfe_contract,
    )
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_daily_ranker import _upside_downside_alignment_metrics

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONTEXT_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    k_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    check(
        "mr13ae_frozen_identity_target_and_architecture",
        (
            profile.name,
            "MR-13AE",
            PREDICTED_SAFETY_CONTEXT_PURE_MFE_TARGET_ID,
            "inception_time_predicted_safety_context_v1",
        ),
        (
            profile.name,
            spec.model_research_id,
            profile.continuous_target_id,
            str(profile.model_architecture),
        ),
    )
    check_true(
        "mr13ae_keeps_exact_mr13k_training_recipe_with_only_one_context_input_change",
        profile.optimizer_name == k_profile.optimizer_name
        and profile.lr_schedule_name == k_profile.lr_schedule_name
        and profile.augmentation_name == k_profile.augmentation_name
        and profile.training_sampling_mode == k_profile.training_sampling_mode
        and profile.training_objective == k_profile.training_objective
        and profile.loss_name == k_profile.loss_name
        and profile.epoch_selection_metric == k_profile.epoch_selection_metric
        and profile.training_label_scope == k_profile.training_label_scope
        and profile.training_sample_scope == k_profile.training_sample_scope
        and get_continuous_ranker_execution_recipe(profile.name).pairwise_reduction
        == get_continuous_ranker_execution_recipe(k_profile.name).pairwise_reduction,
    )
    contract = get_predicted_safety_pure_mfe_contract()
    check_true(
        "mr13ae_target_is_exact_mr13k_pure_mfe_order_without_residualization",
        contract.get("stage2_response") == "canonical_full_horizon_pure_mfe_r"
        and contract.get("stage2_target") == "exact_mr13k_pure_mfe_order_no_residualization"
        and contract.get("stage2_context_used_as_input") is True,
    )
    check(
        "mr13ae_reuses_mr13ad_canonical_predicted_safety_context_owner",
        DAILY_UNIVERSAL_PREDICTED_SAFETY_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        PREDICTED_SAFETY_CONTEXT_OWNER_PROFILE,
    )
    model_spec = get_model_spec("inception_time_predicted_safety_context_v1")
    base_spec = get_model_spec("inception_time_v1")
    check_true(
        "mr13ae_architecture_is_mr13k_inceptiontime_plus_one_direct_predicted_safety_scalar",
        model_spec.inception_depth == base_spec.inception_depth
        and model_spec.inception_filters == base_spec.inception_filters
        and model_spec.pooling == ("global_average", "predicted_safety_percentile_concat")
        and bool(model_spec.use_dataset_context),
    )
    semantics = training_semantics(profile)
    embedded = dict((semantics.get("pairwise_contract") or {}).get("predicted_safety_context_contract") or {})
    check(
        "mr13ae_training_semantics_persist_exact_consumer_contract",
        contract,
        embedded,
    )
    metric_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 10),
            "target_favorable_r": np.arange(1.0, 11.0),
            "target_adverse_r": np.arange(10.0, 0.0, -1.0),
            "target_low_adverse_daily_percentile": np.linspace(0.0, 1.0, 10),
            "target_mfe_daily_percentile": np.linspace(0.0, 1.0, 10),
            "predicted_safety_percentile": np.linspace(0.0, 1.0, 10),
        }
    )
    metric_payload = _upside_downside_alignment_metrics(
        np.arange(10, dtype=np.int64),
        metric_table,
        metric_table["target_favorable_r"].to_numpy(dtype=np.float64)
        - metric_table["target_adverse_r"].to_numpy(dtype=np.float64),
        np.linspace(0.0, 1.0, 10),
    )
    top = dict(metric_payload.get("top_10pct") or {})
    quadrants = dict(top.get("quadrants") or {})
    check_true(
        "mr13ae_standard_sop_common_diagnostic_is_actual_truth_only_with_four_quadrants",
        bool(metric_payload.get("available"))
        and metric_payload.get("status") == "standard_sop_actual_truth_diagnostic_only_no_fit_no_selection"
        and int(metric_payload.get("group_count", 0)) == 10
        and int(top.get("n", 0)) == 1
        and float(top.get("full_mfe_r_mean")) == 10.0
        and float(top.get("adverse_r_mean")) == 1.0
        and float(top.get("low_adverse_r_mean")) == -1.0
        and float(top.get("high_mfe_pct")) == 100.0
        and float(top.get("high_safety_pct")) == 100.0
        and float(top.get("hmhs_pct")) == 100.0
        and float(metric_payload.get("score_to_low_adverse_daily_spearman")) > 0.999
        and set(quadrants) == {"hmhs", "hmls", "lmhs", "lmls"}
        and float((quadrants.get("hmhs") or {}).get("pct")) == 100.0
        and "predicted_safety_to_target_daily_spearman" not in metric_payload
        and "predicted_safety_to_model_score_mean_daily_spearman" not in metric_payload,
    )
    summary["training_performed"] = False
    return results, summary



def validate_breakout_quality_mr13af_high_safety_weighted_pure_mfe_contract_case(_base_params):
    """Keep only MR-13AF historical compatibility and special regression invariants."""

    case_id = "BREAKOUT_QUALITY_MR13AF_HIGH_SAFETY_WEIGHTED_PURE_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_execution_recipe,
        get_continuous_ranker_research_spec,
        get_high_safety_weighted_pure_mfe_contract,
    )
    from filters.breakout_quality.models.spec import get_model_spec
    from services.breakout_quality.train_continuous_ranker import (
        _pairwise_logistic_loss,
        _training_target_for_profile,
    )

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_PREDICTED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    k_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_FULL_HORIZON_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    k_recipe = get_continuous_ranker_execution_recipe(k_profile.name)

    check_true(
        "mr13af_keeps_mr13k_recipe_except_pair_weighting_and_context_coverage",
        profile.optimizer_name == k_profile.optimizer_name
        and profile.lr_schedule_name == k_profile.lr_schedule_name
        and profile.augmentation_name == k_profile.augmentation_name
        and profile.training_sampling_mode == k_profile.training_sampling_mode
        and profile.training_objective == k_profile.training_objective
        and profile.continuous_target_id == k_profile.continuous_target_id
        and profile.loss_name == k_profile.loss_name
        and profile.epoch_selection_metric == k_profile.epoch_selection_metric
        and profile.training_label_scope == k_profile.training_label_scope
        and profile.training_sample_scope == k_profile.training_sample_scope
        and str(profile.model_architecture) == "inception_time_v1"
        and k_recipe.pairwise_reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG
        and recipe.pairwise_reduction == CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    )
    check_true(
        "mr13af_same_target_control_does_not_enable_target_reference_or_checkpoint_reference_eval",
        spec.reference_profile_name is None
        and spec.evaluation_reference_profile_name is None,
    )
    contract = get_high_safety_weighted_pure_mfe_contract()
    check_true(
        "mr13af_contract_has_no_bucket_cutoff_lambda_or_model_input_safety",
        contract.get("stage2_target") == "exact_mr13k_pure_mfe_order_no_residualization"
        and contract.get("stage2_context_used_as_input") is False
        and contract.get("pair_safety_weight")
        == "min(predicted_safety_percentile_i,predicted_safety_percentile_j)"
        and contract.get("pair_weight_combination") == "delta_ndcg_times_min_predicted_safety"
        and contract.get("bucket_or_threshold") is None
        and contract.get("lambda_or_temperature") is None,
    )
    model_spec = get_model_spec("inception_time_v1")
    check_true(
        "mr13af_safety_is_not_network_context",
        not bool(model_spec.use_dataset_context)
        and model_spec.pooling == ("global_average",),
    )

    group_table = pd.DataFrame({
        "predicted_safety_percentile": [0.90, 0.80, 0.10],
    })
    mfe_percentile = np.asarray([1.0, 0.6, 0.1], dtype=np.float32)
    training_target = _training_target_for_profile(
        profile, mfe_percentile.copy(), mfe_percentile.copy(), group_table
    )
    check_true(
        "mr13af_training_target_is_mfe_plus_predicted_safety_for_loss_only",
        training_target.shape == (3, 2)
        and np.allclose(training_target[:, 0], mfe_percentile)
        and np.allclose(training_target[:, 1], [0.90, 0.80, 0.10]),
    )

    dates = np.asarray(["2021-01-04"] * 3)
    margins = torch.tensor([0.3, 0.1, -0.2], dtype=torch.float32, requires_grad=True)
    pure_mfe = torch.tensor(mfe_percentile, dtype=torch.float32)
    all_safe = torch.column_stack([pure_mfe, torch.ones(3, dtype=torch.float32)])
    af_all_safe_loss, af_count = _pairwise_logistic_loss(
        torch, margins, all_safe, dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    )
    af_generic_min_loss, af_generic_count = _pairwise_logistic_loss(
        torch,
        margins,
        all_safe,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_MIN_PREDICTED_SAFETY,
    )
    check_true(
        "mr13af_legacy_combined_reduction_is_numerically_identical_to_generic_min_policy",
        af_count == af_generic_count
        and torch.allclose(
            af_all_safe_loss.detach(), af_generic_min_loss.detach(), atol=0.0, rtol=0.0
        ),
    )

    shortcut_margins = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
    low_unsafe = torch.column_stack([
        pure_mfe,
        torch.tensor([1.0, 0.9, 0.01], dtype=torch.float32),
    ])
    unweighted_loss, _ = _pairwise_logistic_loss(
        torch, shortcut_margins, pure_mfe, dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    weighted_loss, _ = _pairwise_logistic_loss(
        torch, shortcut_margins, low_unsafe, dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_HIGH_SAFETY_MIN_DELTA_NDCG,
    )
    check_true(
        "mr13af_downweights_errors_that_depend_on_very_low_safety_items",
        float(weighted_loss.detach()) < float(unweighted_loss.detach()),
    )

    trainer_source = Path(__file__).resolve().parents[2] / "services" / "breakout_quality" / "train_daily_ranker.py"
    trainer_text = trainer_source.read_text(encoding="utf-8")
    check_true(
        "mr13af_checkpoint_reference_eval_never_falls_back_to_target_comparison_reference",
        "evaluation_reference_profile = research_spec.evaluation_reference_profile_name" in trainer_text
        and "or research_spec.reference_profile_name" not in trainer_text,
    )
    summary["training_performed"] = False
    return results, summary



def validate_breakout_quality_true_hs_scoped_pair_membership_contract_case(_base_params):
    """Protect true-HS Conditional-MFE list membership and lexicographic attribution semantics."""

    case_id = "BREAKOUT_QUALITY_TRUE_HS_SCOPED_PAIR_MEMBERSHIP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    model_spec = get_model_spec(str(profile.model_architecture))
    semantics = training_semantics(profile)
    contract = dict(semantics.get("shared_safety_hs_conditional_mfe_duo_head_contract") or {})

    check_true(
        "true_hs_scope_is_declarative_and_reuses_shared_ak_architecture",
        profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING
        and recipe.training_policy.target_builder
        == CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE
        and recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE
        and recipe.objective_policy.secondary_pair_scope
        == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN
        and float(recipe.objective_policy.secondary_pair_scope_threshold) == 0.50
        and recipe.objective_policy.pair_weight_policy == "none"
        and str(profile.model_architecture) == "inception_time_shared_safety_mfe_v1",
    )
    check_true(
        "true_hs_scope_has_same_gate_ak_attribution_reference_only",
        research.model_gate_reference_profile_name
        == DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        and research.reference_profile_name is None
        and research.evaluation_reference_profile_name is None,
    )
    from config.breakout_quality import get_breakout_quality_workflow_settings
    ao_workflow = get_breakout_quality_workflow_settings(experiment_profile=profile.name)
    ao_is_current = breakout_quality_config.is_breakout_quality_model_test_profile(profile.name)
    check_true(
        "true_hs_historical_forward_gate_follows_current_membership_ssot",
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and ao_workflow.rolling_authorized is ao_is_current
        and ao_workflow.robustness_authorized is ao_is_current,
    )
    check_true(
        "true_hs_contract_is_non_compensatory_and_has_no_safety_context_or_pair_weight",
        contract.get("conditional_mfe_supervision_scope")
        == "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg"
        and contract.get("conditional_mfe_head_inputs")
        == "shared_latent_only_no_predicted_safety_context"
        and contract.get("conditional_mfe_pair_safety_weight") == "none"
        and contract.get("runtime_score")
        == "conditional_mfe_pass_probability_after_predicted_safety_qualification"
        and model_spec.final_mfe_topology_contract().get("mfe_head_inputs")
        == "shared_latent_only_no_safety_prediction_input",
    )

    group_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04"] * 5),
            "target_favorable_r": [8.0, 6.0, 3.0, 2.0, 1.0],
            "target_adverse_r": [4.0, 3.0, 2.0, 1.0, 0.0],
        }
    )
    targets = build_hs_conditional_mfe_targets(
        group_table,
        np.ones(5, dtype=bool),
        true_hs_percentile_cutoff=0.50,
    )
    check_true(
        "true_hs_target_is_full_universe_safety_but_hs_cohort_mfe_percentile",
        np.array_equal(targets.true_hs_mask, np.asarray([False, False, True, True, True]))
        and np.isnan(targets.conditional_mfe_percentile[:2]).all()
        and np.allclose(targets.conditional_mfe_percentile[2:], [1.0, 0.5, 0.0])
        and np.allclose(targets.low_adverse_safety_percentile, [0.0, 0.25, 0.5, 0.75, 1.0]),
    )

    dates = group_table["date"].to_numpy()
    target_tensor = torch.tensor(targets.conditional_mfe_percentile, dtype=torch.float32)
    eligibility = targets.true_hs_mask
    margins_a = torch.tensor([100.0, -100.0, 0.8, 0.1, -0.4], dtype=torch.float32)
    margins_b = torch.tensor([-100.0, 100.0, 0.8, 0.1, -0.4], dtype=torch.float32)
    loss_a, pair_count_a = _pairwise_logistic_loss(
        torch,
        margins_a,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        item_eligibility=eligibility,
    )
    loss_b, pair_count_b = _pairwise_logistic_loss(
        torch,
        margins_b,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        item_eligibility=eligibility,
    )
    check_true(
        "ls_margin_changes_cannot_change_hs_sublist_delta_ndcg_geometry",
        pair_count_a == pair_count_b == 3
        and torch.allclose(loss_a.detach(), loss_b.detach(), atol=0.0, rtol=0.0),
    )

    from services.breakout_quality.train_daily_ranker import _hs_lexicographic_reference_control
    loader_group_table = group_table.assign(
        ticker=["A", "B", "C", "D", "E"],
        label=np.ones(5, dtype=np.int64),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        frozen_path = Path(tmpdir) / "ak_forward_scores.csv"
        pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "date": ["2021-01-04"] * 5,
                "group_index": [0, 1, 2, 3, 4],
                "raw_safety_score": [0.10, 0.20, 0.30, 0.80, 0.90],
                "raw_mfe_score": [0.90, 0.80, 0.70, 0.60, 0.50],
            }
        ).to_csv(frozen_path, index=False)
        with patch(
            "services.breakout_quality.train_daily_ranker.resolve_continuous_ranker_oos_score_path",
            return_value=frozen_path,
        ):
            reference_control = _hs_lexicographic_reference_control(
                filter_id="breakout_quality_v1",
                reference_profile_name=(
                    DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
                ),
                group_table=loader_group_table,
                hs_targets=targets,
                oos_ids=np.arange(5, dtype=np.int64),
                breakout_candidate_ids=np.asarray([2, 3, 4], dtype=np.int64),
            )
    check_true(
        "ak_same_gate_reference_uses_daily_universal_safety_percentile_before_breakout_filter",
        reference_control.get("available") is True
        and reference_control["oos"]["lexicographic_model_gate"]["predicted_hs_n"] == 3
        and reference_control["breakout_candidate_oos"]["lexicographic_model_gate"][
            "predicted_safety_percentile_source"
        ]
        == "caller_supplied_daily_universal_percentile_no_subset_rerank",
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_hs_qualification_conditional_mfe_contract_case(_base_params):
    """Protect direct HS qualification while reusing AO true-HS Conditional-MFE geometry."""

    case_id = "BREAKOUT_QUALITY_HS_QUALIFICATION_CONDITIONAL_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_breakout_quality_workflow_settings,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import (
        _binary_threshold_pair_target,
        _pairwise_logistic_loss,
        hs_conditional_mfe_metrics,
    )

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    ao_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    ao_recipe = get_continuous_ranker_execution_recipe(ao_profile.name)
    semantics = training_semantics(profile)
    contract = dict(
        semantics.get("shared_hs_qualification_conditional_mfe_duo_head_contract") or {}
    )

    check_true(
        "hs_qualification_changes_only_ao_primary_head_supervision_semantic",
        research.model_research_id == "MR-13AR"
        and profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING
        and recipe.training_policy.target_builder
        == ao_recipe.training_policy.target_builder
        == CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE
        and recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE
        and recipe.objective_policy.secondary_pair_scope
        == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN
        and float(recipe.objective_policy.secondary_pair_scope_threshold) == 0.50
        and recipe.objective_policy.pair_weight_policy == "none",
    )
    check_true(
        "hs_qualification_keeps_ao_architecture_epoch_selection_and_same_gate_reference",
        str(profile.model_architecture) == str(ao_profile.model_architecture)
        == "inception_time_shared_safety_mfe_v1"
        and profile.epoch_selection_metric == ao_profile.epoch_selection_metric
        == "hs_conditional_mfe_mean_daily_spearman"
        and research.model_gate_reference_profile_name == ao_profile.name,
    )
    check_true(
        "hs_qualification_contract_is_binary_boundary_plus_true_hs_upside",
        contract.get("qualification_target")
        == "indicator_of_same_date_low_adverse_safety_percentile_gte_0.50"
        and contract.get("qualification_pair_scope")
        == "same_date_hs_vs_ls_only_same_cohort_ties_excluded"
        and contract.get("conditional_mfe_supervision_scope")
        == "true_hs_items_only_sublist_before_rank_positions_idcg_and_delta_ndcg"
        and contract.get("conditional_mfe_head_inputs")
        == "shared_latent_only_no_predicted_safety_context"
        and contract.get("conditional_mfe_pair_safety_weight") == "none",
    )
    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile.name)
    is_current = breakout_quality_config.is_breakout_quality_model_test_profile(profile.name)
    check_true(
        "hs_qualification_historical_forward_gate_follows_current_membership_ssot",
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and workflow.rolling_authorized is is_current
        and workflow.robustness_authorized is is_current,
    )

    group_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04"] * 5),
            "target_favorable_r": [8.0, 6.0, 3.0, 2.0, 1.0],
            "target_adverse_r": [4.0, 3.0, 2.0, 1.0, 0.0],
            "label": [1, 1, 1, 1, 1],
        }
    )
    targets = build_hs_conditional_mfe_targets(
        group_table, np.ones(5, dtype=bool), true_hs_percentile_cutoff=0.50
    )
    safety_continuous = torch.tensor(targets.low_adverse_safety_percentile, dtype=torch.float32)
    safety_binary = _binary_threshold_pair_target(torch, safety_continuous, 0.50)
    check_true(
        "hs_qualification_binary_target_ties_same_cohort_and_preserves_true_hs_boundary",
        np.array_equal(targets.true_hs_mask, [False, False, True, True, True])
        and torch.equal(safety_binary, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])),
    )

    dates = group_table["date"].to_numpy()
    safety_margins = torch.tensor([-0.9, -0.5, 0.1, 0.6, 0.9], dtype=torch.float32)
    binary_loss, binary_pairs = _pairwise_logistic_loss(
        torch,
        safety_margins,
        safety_binary,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    continuous_loss, continuous_pairs = _pairwise_logistic_loss(
        torch,
        safety_margins,
        safety_continuous,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    conditional_loss, conditional_pairs = _pairwise_logistic_loss(
        torch,
        torch.tensor([100.0, -100.0, 0.9, 0.3, -0.2], dtype=torch.float32),
        torch.tensor(targets.conditional_mfe_percentile, dtype=torch.float32),
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        item_eligibility=targets.true_hs_mask,
    )
    check_true(
        "hs_qualification_primary_has_only_hs_vs_ls_pairs_while_secondary_keeps_ao_hs_sublist",
        binary_loss is not None
        and continuous_loss is not None
        and conditional_loss is not None
        and binary_pairs == 6
        and continuous_pairs == 10
        and conditional_pairs == 3,
    )

    metric_scores = {
        "raw_safety": np.asarray([-0.8, -0.4, 0.2, 0.6, 0.9], dtype=np.float32),
        "conditional_mfe": np.asarray([5.0, 4.0, 0.9, 0.4, 0.1], dtype=np.float32),
    }
    metric_payload = hs_conditional_mfe_metrics(
        np.arange(5, dtype=np.int64),
        group_table,
        targets,
        metric_scores,
        include_top_k_quality=False,
    )
    check_true(
        "hs_qualification_metrics_measure_boundary_purity_without_changing_conditional_mfe_truth",
        metric_payload["hs_qualification"]["pairwise_concordance"] == 1.0
        and metric_payload["lexicographic_model_gate"]["predicted_hs_true_ls_pct"] == 0.0
        and metric_payload["lexicographic_model_gate"]["true_hs_recall_pct"] == 100.0
        and metric_payload["conditional_mfe_true_hs"]["pairwise_concordance"] == 1.0,
    )

    from services.breakout_quality.train_daily_ranker import _hs_lexicographic_reference_control
    loader_group_table = group_table.assign(ticker=["A", "B", "C", "D", "E"])
    with tempfile.TemporaryDirectory() as tmpdir:
        frozen_path = Path(tmpdir) / "ao_forward_scores.csv"
        pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "date": ["2021-01-04"] * 5,
                "group_index": [0, 1, 2, 3, 4],
                "raw_safety_score": [0.10, 0.20, 0.30, 0.80, 0.90],
                "conditional_mfe_score": [0.10, 0.20, 0.30, 0.80, 0.90],
                "model_score": [0.90, 0.80, 0.70, 0.20, 0.10],
            }
        ).to_csv(frozen_path, index=False)
        with patch(
            "services.breakout_quality.train_daily_ranker.resolve_continuous_ranker_oos_score_path",
            return_value=frozen_path,
        ):
            reference_control = _hs_lexicographic_reference_control(
                filter_id="breakout_quality_v1",
                reference_profile_name=ao_profile.name,
                group_table=loader_group_table,
                hs_targets=targets,
                oos_ids=np.arange(5, dtype=np.int64),
                breakout_candidate_ids=np.arange(5, dtype=np.int64),
            )
    check_true(
        "hs_qualification_ao_reference_uses_conditional_mfe_primary_column_not_generic_model_score",
        reference_control.get("available") is True
        and reference_control.get("reference_ranking_head") == "conditional_mfe"
        and reference_control.get("mfe_score_column") == "conditional_mfe_score"
        and reference_control["oos"]["lexicographic_model_gate"]["selected_high_mfe_pct"]
        != reference_control["oos"]["lexicographic_model_gate"]["selected_hmls_pct"],
    )

    summary["training_performed"] = False
    return results, summary



def validate_breakout_quality_hs_boundary_weighted_conditional_mfe_contract_case(_base_params):
    """Protect P50-boundary-focused HS qualification while preserving AR Conditional-MFE."""

    case_id = "BREAKOUT_QUALITY_HS_BOUNDARY_WEIGHTED_CONDITIONAL_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE,
        BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
        BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_breakout_quality_workflow_settings,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY,
        CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE,
        get_continuous_ranker_primary_pair_weight_policy,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_conditional_mfe_targets
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import (
        _binary_threshold_pair_target,
        _pairwise_logistic_loss,
        hs_conditional_mfe_metrics,
    )

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    ar_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    ar_recipe = get_continuous_ranker_execution_recipe(ar_profile.name)
    semantics = training_semantics(profile)
    contract = dict(
        semantics.get("shared_hs_qualification_conditional_mfe_duo_head_contract") or {}
    )

    check_true(
        "hs_boundary_weighted_changes_only_ar_primary_truth_side_pair_weight",
        research.model_research_id == "MR-13AS"
        and profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SHARED_HS_BOUNDARY_WEIGHTED_QUALIFICATION_CONDITIONAL_MFE_PAIRWISE_RANKING
        and recipe.training_policy.target_builder == ar_recipe.training_policy.target_builder
        and recipe.training_policy.loss_handler == ar_recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_HS_QUALIFICATION_SCOPED_MFE_DUO_PAIRWISE
        and recipe.objective_policy.secondary_pair_scope
        == ar_recipe.objective_policy.secondary_pair_scope
        == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN
        and float(recipe.objective_policy.secondary_pair_scope_threshold) == 0.50
        and get_continuous_ranker_primary_pair_weight_policy(profile.training_objective)
        == CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY
        and get_continuous_ranker_primary_pair_weight_policy(ar_profile.training_objective)
        == CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_NONE,
    )
    check_true(
        "hs_boundary_weighted_semantics_are_truth_side_only_and_parameter_free",
        contract.get("qualification_pair_weighting")
        == CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY
        and contract.get("qualification_pair_weight_formula")
        == "1_minus_abs_same_date_safety_percentile_pair_gap"
        and contract.get("qualification_pair_weight_role")
        == "truth_side_supervision_only_no_model_input_no_direction_change"
        and research.model_gate_reference_profile_name == ar_profile.name,
    )
    check_true(
        "hs_boundary_weighted_historical_identity_does_not_pin_current_membership",
        research.model_research_id == "MR-13AS"
        and tuple(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE) in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
        and all(
            pair in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
            for pair in BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES
        )
        and len(BREAKOUT_QUALITY_MODEL_TEST_PROFILES)
        == len(BREAKOUT_QUALITY_MODEL_TEST_REFERENCE_PROFILES) + 1,
    )
    check_true(
        "hs_boundary_weighted_historical_forward_gate_does_not_pin_current_test_membership",
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and not breakout_quality_config.is_breakout_quality_model_test_profile(profile.name)
        and breakout_quality_config.is_breakout_quality_model_test_profile(
            BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[1]
        ),
    )

    dates = pd.to_datetime(["2021-01-04"] * 4).to_numpy()
    safety_truth = torch.tensor([0.10, 0.49, 0.51, 0.90], dtype=torch.float32)
    binary_target = _binary_threshold_pair_target(torch, safety_truth, 0.50)
    margins = torch.tensor([-1.5, 0.7, -0.7, 1.5], dtype=torch.float32)
    unweighted_loss, unweighted_pairs = _pairwise_logistic_loss(
        torch,
        margins,
        binary_target,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    boundary_loss, boundary_pairs = _pairwise_logistic_loss(
        torch,
        margins,
        binary_target,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_truth_weight_values=safety_truth,
        pair_truth_weight_policy=CONTINUOUS_RANKER_PRIMARY_PAIR_WEIGHT_POLICY_BINARY_BOUNDARY_PROXIMITY,
    )
    check_true(
        "hs_boundary_weighting_preserves_binary_pair_direction_and_pair_membership",
        unweighted_loss is not None
        and boundary_loss is not None
        and unweighted_pairs == boundary_pairs == 4
        and not torch.allclose(unweighted_loss.detach(), boundary_loss.detach(), atol=1e-8, rtol=0.0),
    )

    group_count = 21
    group_table = pd.DataFrame({
        "date": pd.to_datetime(["2021-01-04"] * group_count),
        "target_favorable_r": np.linspace(0.2, 4.2, group_count),
        # Descending adverse produces ascending Low-Adverse safety percentiles 0..1.
        "target_adverse_r": np.linspace(4.0, 0.0, group_count),
        "label": np.ones(group_count, dtype=np.int64),
    })
    targets = build_hs_conditional_mfe_targets(
        group_table, np.ones(group_count, dtype=bool), true_hs_percentile_cutoff=0.50
    )
    metric_scores = {
        "raw_safety": np.asarray(targets.low_adverse_safety_percentile, dtype=np.float32),
        "conditional_mfe": np.nan_to_num(
            np.asarray(targets.conditional_mfe_percentile, dtype=np.float32), nan=-1.0
        ),
    }
    metric_payload = hs_conditional_mfe_metrics(
        np.arange(group_count, dtype=np.int64),
        group_table,
        targets,
        metric_scores,
        include_top_k_quality=False,
        top_k=3,
    )
    gate = dict(metric_payload["lexicographic_model_gate"])
    oracle = dict(metric_payload["true_hs_oracle_gate"])
    check_true(
        "hs_boundary_metrics_cover_p50_bands_and_true_hs_oracle_gap_without_reranking_truth",
        metric_payload["hs_qualification_boundary"]["p40_p60"]["pairwise_concordance"] == 1.0
        and metric_payload["hs_qualification_boundary"]["p45_p55"]["pairwise_concordance"] == 1.0
        and gate["predicted_hs_true_ls_pct"] == 0.0
        and gate["true_hs_recall_pct"] == 100.0
        and gate["hmhs_gap_vs_true_hs_oracle_pp"] == 0.0
        and gate["high_mfe_gap_vs_true_hs_oracle_pp"] == 0.0
        and oracle["selected_hmls_pct"] == 0.0,
    )

    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_hs_priority_mfe_contract_case(_base_params):
    """Protect all-daily HS-priority ranking truth and direct-score semantics."""

    case_id = "BREAKOUT_QUALITY_HS_PRIORITY_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_breakout_quality_workflow_settings,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
        CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_priority_mfe_targets
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    model_spec = get_model_spec(str(profile.model_architecture))
    semantics = training_semantics(profile)
    contract = dict(semantics.get("shared_safety_hs_priority_mfe_duo_head_contract") or {})

    check_true(
        "hs_priority_recipe_uses_all_items_without_safety_pair_weight_or_gate",
        research.model_research_id == "MR-13AP"
        and profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_MFE_PAIRWISE_RANKING
        and recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE
        and recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE
        and recipe.objective_policy.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
        and recipe.objective_policy.secondary_pair_scope_threshold is None
        and recipe.objective_policy.pair_weight_policy == "none"
        and research.model_gate_reference_profile_name is None,
    )
    check_true(
        "hs_priority_reuses_independent_shared_architecture_and_direct_final_head",
        str(profile.model_architecture) == "inception_time_shared_safety_mfe_v1"
        and model_spec.final_mfe_topology_contract().get("mfe_head_inputs")
        == "shared_latent_only_no_safety_prediction_input"
        and contract.get("priority_head_inputs") == "shared_latent_only_no_predicted_safety_context"
        and contract.get("priority_pair_safety_weight") == "none"
        and contract.get("runtime_score") == "hs_priority_mfe_pass_probability_direct_all_daily_ranking",
    )
    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile.name)
    is_current = breakout_quality_config.is_breakout_quality_model_test_profile(profile.name)
    check_true(
        "hs_priority_historical_forward_gate_follows_current_membership_ssot",
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and workflow.rolling_authorized is is_current
        and workflow.robustness_authorized is is_current,
    )

    group_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04"] * 5),
            # First two rows are deliberately huge-MFE LS examples.
            "target_favorable_r": [8.0, 6.0, 3.0, 2.0, 1.0],
            "target_adverse_r": [4.0, 3.0, 2.0, 1.0, 0.0],
            "label": [1, 1, 1, 1, 1],
        }
    )
    targets = build_hs_priority_mfe_targets(group_table, np.ones(5, dtype=bool))
    check_true(
        "hs_priority_truth_puts_every_ls_at_zero_and_orders_only_hs_by_mfe",
        np.array_equal(targets.true_hs_mask, np.asarray([False, False, True, True, True]))
        and np.allclose(targets.low_adverse_safety_percentile, [0.0, 0.25, 0.5, 0.75, 1.0])
        and np.isnan(targets.conditional_mfe_percentile[:2]).all()
        and np.allclose(targets.conditional_mfe_percentile[2:], [1.0, 0.5, 0.0])
        and np.allclose(targets.hs_priority_mfe_relevance, [0.0, 0.0, 1.0, 0.75, 0.5]),
    )
    check_true(
        "hs_priority_truth_is_non_compensatory_even_for_extreme_hmls",
        float(targets.hs_priority_mfe_relevance[0]) == 0.0
        and float(targets.hs_priority_mfe_relevance[1]) == 0.0
        and float(np.min(targets.hs_priority_mfe_relevance[targets.true_hs_mask]))
        > float(np.max(targets.hs_priority_mfe_relevance[~targets.true_hs_mask])),
    )

    dates = group_table["date"].to_numpy()
    margins = torch.tensor([0.9, 0.7, 0.8, 0.1, -0.4], dtype=torch.float32)
    target_tensor = torch.tensor(targets.hs_priority_mfe_relevance, dtype=torch.float32)
    full_loss, full_pair_count = _pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    )
    hs_only_loss, hs_only_pair_count = _pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        item_eligibility=targets.true_hs_mask,
    )
    check_true(
        "hs_priority_full_list_adds_hs_vs_ls_boundary_pairs_while_ls_vs_ls_stays_tied",
        full_loss is not None
        and hs_only_loss is not None
        and full_pair_count == 9
        and hs_only_pair_count == 3,
    )

    from services.breakout_quality.train_continuous_ranker import hs_priority_mfe_metrics
    metric_scores = {
        "raw_safety": np.asarray([0.0, 0.2, 0.5, 0.7, 0.9], dtype=np.float32),
        "conditional_mfe": np.asarray([0.1, 0.2, 0.9, 0.7, 0.5], dtype=np.float32),
    }
    metric_payload = hs_priority_mfe_metrics(
        np.arange(5, dtype=np.int64), group_table, targets, metric_scores,
        include_top_k_quality=False,
    )
    check_true(
        "hs_priority_metrics_separate_boundary_and_hs_only_upside_diagnostics",
        metric_payload["hs_priority_mfe"]["pairwise_concordance"] == 1.0
        and metric_payload["hs_vs_ls_boundary"]["pairwise_concordance"] == 1.0
        and metric_payload["conditional_mfe_true_hs"]["pairwise_concordance"] == 1.0,
    )

    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_hs_priority_stratified_mfe_contract_case(_base_params):
    """Protect pair-stratified normalization while preserving MR-13AP truth/geometry."""

    case_id = "BREAKOUT_QUALITY_HS_PRIORITY_STRATIFIED_MFE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import numpy as np
    import pandas as pd
    import torch
    from config.breakout_quality import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING,
        get_breakout_quality_experiment_profile,
        get_breakout_quality_workflow_settings,
        get_continuous_ranker_research_spec,
    )
    from config.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_STRATIFIED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_CROSS,
        CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_WITHIN_POSITIVE,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL,
        CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE,
    )
    from config.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality.hs_conditional_mfe import build_hs_priority_mfe_targets
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality.train_continuous_ranker import (
        _binary_partition_stratified_pairwise_loss,
        _pairwise_logistic_loss,
    )

    profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    research = get_continuous_ranker_research_spec(profile.name)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    semantics = training_semantics(profile)
    contract = dict(
        semantics.get("shared_safety_hs_priority_stratified_mfe_duo_head_contract") or {}
    )

    check_true(
        "hs_priority_stratified_recipe_changes_only_secondary_pair_aggregation",
        research.model_research_id == "MR-13AQ"
        and profile.training_objective
        == TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_PRIORITY_STRATIFIED_MFE_PAIRWISE_RANKING
        and recipe.training_policy.target_builder == CONTINUOUS_RANKER_TARGET_BUILDER_HS_PRIORITY_MFE
        and recipe.training_policy.loss_handler
        == CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_STRATIFIED_MFE_DUO_PAIRWISE
        and recipe.objective_policy.secondary_pair_scope == CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_ALL
        and recipe.objective_policy.secondary_pair_scope_threshold is None
        and recipe.objective_policy.pair_weight_policy == "none",
    )
    check_true(
        "hs_priority_stratified_contract_keeps_ap_truth_and_full_list_geometry",
        contract.get("priority_mfe_target")
        == "mr13ap_truth_ls_equals_0_else_0.5_plus_0.5_times_same_date_mfe_percentile_within_true_hs"
        and contract.get("priority_stratum_geometry")
        == "same_full_list_predicted_rank_positions_idcg_and_delta_ndcg"
        and contract.get("priority_stratum_normalization")
        == "each_stratum_normalized_by_own_delta_ndcg_weight_sum_then_fixed_equal_mean"
        and contract.get("priority_pair_safety_weight") == "none",
    )
    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile.name)
    is_current = breakout_quality_config.is_breakout_quality_model_test_profile(profile.name)
    check_true(
        "hs_priority_stratified_historical_forward_gate_follows_current_membership_ssot",
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and workflow.rolling_authorized is is_current
        and workflow.robustness_authorized is is_current,
    )

    group_table = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-04"] * 5),
            "target_favorable_r": [8.0, 6.0, 3.0, 2.0, 1.0],
            "target_adverse_r": [4.0, 3.0, 2.0, 1.0, 0.0],
            "label": [1, 1, 1, 1, 1],
        }
    )
    aq_targets = build_hs_priority_mfe_targets(group_table, np.ones(5, dtype=bool))
    ap_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_PRIORITY_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    check_true(
        "hs_priority_stratified_reuses_exact_ap_target_builder_and_truth",
        get_continuous_ranker_execution_recipe(ap_profile.name).training_policy.target_builder
        == recipe.training_policy.target_builder
        and np.allclose(aq_targets.hs_priority_mfe_relevance, [0.0, 0.0, 1.0, 0.75, 0.5])
        and np.array_equal(aq_targets.true_hs_mask, [False, False, True, True, True]),
    )

    dates = group_table["date"].to_numpy()
    margins = torch.tensor([0.9, 0.7, 0.8, 0.1, -0.4], dtype=torch.float32)
    target_tensor = torch.tensor(aq_targets.hs_priority_mfe_relevance, dtype=torch.float32)
    membership = torch.tensor(aq_targets.true_hs_mask, dtype=torch.bool)
    boundary_loss, boundary_pairs = _pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_partition_membership=membership,
        pair_partition_relation=CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_CROSS,
    )
    within_hs_loss, within_hs_pairs = _pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        pair_partition_membership=membership,
        pair_partition_relation=CONTINUOUS_RANKER_PAIR_PARTITION_RELATION_WITHIN_POSITIVE,
    )
    combined_loss, combined_boundary_pairs, combined_within_pairs = (
        _binary_partition_stratified_pairwise_loss(
            torch,
            margins,
            target_tensor,
            dates,
            membership,
            reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        )
    )
    check_true(
        "hs_priority_stratified_pair_partition_is_six_boundary_plus_three_hs_within",
        boundary_pairs == combined_boundary_pairs == 6
        and within_hs_pairs == combined_within_pairs == 3,
    )
    check_true(
        "hs_priority_stratified_loss_is_equal_mean_after_each_stratum_normalization",
        boundary_loss is not None
        and within_hs_loss is not None
        and combined_loss is not None
        and torch.allclose(
            combined_loss.detach(),
            (0.5 * (boundary_loss + within_hs_loss)).detach(),
            atol=0.0,
            rtol=0.0,
        ),
    )

    hs_sublist_loss, hs_sublist_pairs = _pairwise_logistic_loss(
        torch,
        margins,
        target_tensor,
        dates,
        reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        item_eligibility=membership,
    )
    check_true(
        "hs_priority_stratified_within_hs_filter_preserves_full_list_ndcg_not_hs_sublist_geometry",
        hs_sublist_pairs == within_hs_pairs == 3
        and hs_sublist_loss is not None
        and within_hs_loss is not None
        and not torch.allclose(
            hs_sublist_loss.detach(), within_hs_loss.detach(), atol=1e-8, rtol=0.0
        ),
    )

    summary["training_performed"] = False
    return results, summary

