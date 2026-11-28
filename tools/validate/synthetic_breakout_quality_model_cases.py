from __future__ import annotations

from dataclasses import replace

import core.breakout_quality_registry as breakout_quality_registry
import core.breakout_quality_policy as breakout_quality_policy
from .checks import (
    raises_expected,
    bind_synthetic_case,
    run_bound_checks,
)

from config import breakout_quality as breakout_quality_config

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
    Path,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR,
    CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
    get_continuous_ranker_research_spec,
    DAILY_FULL_HORIZON_OPPORTUNITY_TARGET_ID,
    DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SimpleNamespace,
    StrategyAlignedContinuousTargetSpec,
    TARGET_MANIFEST_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    ast,
    build_daily_full_horizon_opportunity_contract,
    build_daily_full_horizon_pure_mfe_contract,
    build_daily_percentile_targets,
    build_file_manifest,
    build_strategy_aligned_group_targets,
    continuous_ranker_scope_group_ids,
    compute_daily_opportunity_target_batch,
    get_breakout_quality_experiment_profile,
    json,
    load_validated_continuous_target_arrays,
    math,
    np,
    patch,
    pd,
    resolve_continuous_target_dir,
    daily_full_horizon_opportunity_target_from_cached_path,
    daily_full_horizon_pure_mfe_target_from_cached_path,
    daily_opportunity_no_time_target_from_cached_path,
    strategy_aligned_target_from_cached_path,
    tempfile,
)

from .source_index import read_source_ast, read_source_import_nodes, read_source_text


def _start_case(case_id: str):
    results, summary, check, check_true = bind_synthetic_case(case_id, "synthetic_breakout_quality")
    return results, summary, check, check_true




def _finish_case(results, summary):
    summary["training_performed"] = False
    return results, summary






_CONTROLLED_PROFILE_RECIPE_FIELDS = (
    "optimizer_name",
    "lr_schedule_name",
    "augmentation_name",
    "training_sampling_mode",
    "training_objective",
    "loss_name",
    "epoch_selection_metric",
    "training_label_scope",
    "training_sample_scope",
)
















def _historical_forward_gate_matches_current_membership(profile, research) -> bool:
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings

    workflow = get_breakout_quality_workflow_settings(experiment_profile=profile.name)
    is_current = breakout_quality_policy.is_breakout_quality_model_test_profile(profile.name)
    return bool(
        research.selection_pit_authorized is False
        and research.current_time_validation_authorized is False
        and workflow.rolling_authorized is is_current
        and workflow.robustness_authorized is is_current
    )


def _five_item_hs_group_table(*, include_label: bool):
    data = {
        "date": pd.to_datetime(["2021-01-04"] * 5),
        "target_favorable_r": [8.0, 6.0, 3.0, 2.0, 1.0],
        "target_adverse_r": [4.0, 3.0, 2.0, 1.0, 0.0],
    }
    if include_label:
        data["label"] = [1] * 5
    return pd.DataFrame(data)




def validate_breakout_quality_continuous_target_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_TARGET"
    results, summary, check, check_true = _start_case(case_id)

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    contract = spec.contract_payload()
    run_bound_checks(
        check,
        (
            ('continuous_target_id_is_versioned', STRATEGY_ALIGNED_TARGET_ID, contract['target_id'],),
            ('continuous_target_uses_no_split_or_oos_parameters', (False, False, 'none', 'none'), (contract['split_derived_parameters'], contract['oos_derived_parameters'], contract['normalization'], contract['clipping']),),
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
    run_bound_checks(
        check_true,
        (
            ('prepare_continuous_target_routes_only_to_canonical_service_builder', bool('services.breakout_quality.continuous_target_builder' in prepare_source and 'tools.audit.breakout_quality.continuous_target' not in prepare_source and ('approved-workflow-rebuild' not in prepare_source)),),
            ('retired_11c_to_11f_target_audit_implementations_are_not_runtime_or_catalog_commands', bool(all((not path.exists() for path in retired_paths)) and 'audit-continuous-target' not in catalog_source and ('audit-qualified-candidate-set' not in catalog_source) and ('audit-target-attribution' not in catalog_source) and ('audit-target-time-ablation' not in catalog_source) and ('audit-no-time-target' not in catalog_source) and ('module="services.breakout_quality.point_in_time_audit"' in catalog_source)),),
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
    results, summary, check, check_true = _start_case(case_id)

    profile = get_breakout_quality_experiment_profile(STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE)

    # Capability-oriented contract: every registered continuous-ranker profile is
    # automatically checked here.  A new experiment that only recombines existing
    # target/context/objective capabilities must not require a new synthetic case.
    import config.breakout_quality as breakout_quality_config
    import core.breakout_quality_registry as breakout_quality_registry
    import core.breakout_quality_policy as breakout_quality_policy
    import core.breakout_quality_runtime as continuous_ranker_runtime
    import core.breakout_quality_runtime_resolver as continuous_ranker_runtime_resolver
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_CONTEXT_ROLE_COVERAGE,
        CONTINUOUS_RANKER_CONTEXT_ROLE_PAIR_WEIGHT,
        CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        CONTINUOUS_RANKER_CONTEXT_SOURCE_PREDICTED_SAFETY,
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
    from core.breakout_quality_runtime_resolver import (
        get_continuous_ranker_execution_recipe,
    )
    from filters.breakout_quality.artifact_dependency_registry import (
        ARTIFACT_CONTINUOUS_TARGET,
        ARTIFACT_DATASET_CORE,
        ARTIFACT_TRADE_COUNT_SOURCE,
        required_upstream_artifact_types,
    )
    from filters.breakout_quality.models.architectures import get_architecture_descriptor
    from filters.breakout_quality.predicted_context_artifact import (
        maybe_predicted_context_artifact_spec,
    )
    from filters.breakout_quality.ranker_training_contract import training_semantics

    check(
        "continuous_ranker_runtime_contract_has_dedicated_owner",
        "core.breakout_quality_runtime",
        continuous_ranker_runtime.ContinuousRankerExecutionRecipe.__module__,
    )
    check(
        "continuous_ranker_profile_enabled_objective_membership_derives_from_composition_registry",
        tuple(breakout_quality_registry.CONTINUOUS_RANKER_TRAINING_OBJECTIVES),
        tuple(get_profile_enabled_continuous_ranker_training_objectives()),
        note=(
            "config must not maintain a second hand-written objective membership list; "
            "profile eligibility is declared by composition profile_loss_metrics"
        ),
    )
    check_true(
        "continuous_ranker_recipe_resolver_reexports_canonical_config_resolver",
        breakout_quality_policy.get_continuous_ranker_execution_recipe
        is continuous_ranker_runtime_resolver.get_continuous_ranker_execution_recipe,
        note=(
            "the acyclic runtime resolver gateway must forward the single canonical "
            "profile-to-recipe resolver owned by core.breakout_quality_policy"
        ),
    )
    runtime_public_names = set(continuous_ranker_runtime.__all__)
    runtime_facade_bypass_imports = []
    project_root = Path(__file__).resolve().parents[2]
    for source_root_name in ("filters", "services"):
        for source_path in sorted((project_root / source_root_name).rglob("*.py")):
            # Most runtime files do not import this config facade.  A text
            # prefilter is false-positive-safe and avoids parsing unrelated
            # modules; matching candidates still use the canonical AST import
            # index, so the contract semantics are unchanged.
            if "config.breakout_quality" not in read_source_text(source_path):
                continue
            for node in read_source_import_nodes(source_path):
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
            "core.breakout_quality_runtime and the profile resolver from "
            "core.breakout_quality_runtime_resolver; config.breakout_quality is not a "
            "generic-consumer runtime import surface"
        ),
    )

    # Generic regression for migrated owner boundaries: Python accepts a stale
    # from-import syntactically, but fails only when that downstream path runs.
    # Validate all registered-name imports without executing a data producer.
    unresolved_registry_imports = []
    for source_root_name in ("filters", "services"):
        for source_path in sorted((project_root / source_root_name).rglob("*.py")):
            if "core.breakout_quality_registry" not in read_source_text(source_path):
                continue
            for node in read_source_import_nodes(source_path):
                if not isinstance(node, ast.ImportFrom) or node.module != "core.breakout_quality_registry":
                    continue
                missing = sorted(
                    alias.name for alias in node.names
                    if alias.name != "*" and not hasattr(breakout_quality_registry, alias.name)
                )
                if missing:
                    unresolved_registry_imports.append(
                        f"{source_path.relative_to(project_root).as_posix()}:"
                        + ",".join(missing)
                    )
    check(
        "continuous_ranker_registry_from_import_symbols_exist",
        [],
        unresolved_registry_imports,
        note="generic consumers must import recipe resolution from its current canonical owner",
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
        synthetic_contract = breakout_quality_registry.get_predicted_safety_pair_weight_contract(
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
            "owned by core.breakout_quality_runtime; filters/services consume resolved "
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
        if (
            registered_profile.model_architecture is not None
            and get_architecture_descriptor(registered_profile.model_architecture).has_capability("lagged_trade_count_input")
        ):
            expected_dependencies.append(ARTIFACT_TRADE_COUNT_SOURCE)
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
                    breakout_quality_registry.get_predicted_safety_pair_weight_contract(
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

    run_bound_checks(
        check,
        (
            ('continuous_ranker_all_registered_profiles_share_generic_runtime_contract', [], registered_runtime_failures,),
            ('continuous_ranker_generic_contract_covers_every_registered_profile', len(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES), len(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES) - len(registered_runtime_failures),),
        ),
    )

    registry_source = read_source_text("core/breakout_quality_registry.py")
    trainer_source = read_source_text(
        "services/breakout_quality/train_continuous_ranker.py"
    )
    training_contract_source = read_source_text(
        "filters/breakout_quality/ranker_training_contract.py"
    )
    check_true(
        "continuous_ranker_profile_and_research_spec_delegate_objective_legality_to_composition_owner",
        "expected_loss = {" not in registry_source
        and "scoped secondary pair supervision只允許支援該capability的training objective"
        not in registry_source
        and ".validate_profile_loss_metric(" in registry_source
        and ".validate_research_spec(" in registry_source,
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

    # Validate the current 2-logit classifier and pass probability with a real forward pass.
    from filters.breakout_quality.models.architectures import (
        INCEPTION_TIME_V1,
    )
    from filters.breakout_quality.models.factory import build_model
    from filters.breakout_quality.models.spec import get_model_spec
    from filters.breakout_quality.models.runtime import require_torch

    _torch, _nn = require_torch()
    _old_spec = get_model_spec(INCEPTION_TIME_V1)
    # Keep model initialization from advancing later synthetic cases' RNG state.
    with _torch.random.fork_rng(devices=[]):
        _old_model = build_model(10, 0, architecture=INCEPTION_TIME_V1).eval()
        with _torch.inference_mode():
            _feature_probe = _torch.zeros(2, 300, 10)
            _old_logits = _old_model(_feature_probe, None)
            _old_probability = _torch.softmax(_old_logits.float(), dim=1)[:, LABEL_PASS]
    check(
        "continuous_ranker_reuses_active_two_logit_head_and_pass_probability",
        (True, True, True),
        (
            _old_spec.classifier_output_classes is None
            and _old_model.classifier.out_features == 2
            and tuple(_old_logits.shape) == (2, 2)
            and bool(_torch.all((_old_probability >= 0) & (_old_probability <= 1)).item()),
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

    # Canonical registry resolution is part of the public Research contract.  The
    # synthetic case therefore selects an existing model-gate-only profile instead
    # of inventing an experiment identity that production code is required to reject.
    model_gate_profile = next(
        profile_name
        for profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        if not get_continuous_ranker_research_spec(profile_name).selection_pit_authorized
        and not breakout_quality_policy.get_breakout_quality_workflow_settings(
            experiment_profile=profile_name
        ).rolling_authorized
    )
    model_gate_settings = breakout_quality_policy.get_breakout_quality_workflow_settings(
        experiment_profile=model_gate_profile
    )
    ready_plan = SimpleNamespace(blocked=False, actions=(), overall_status="READY")
    gate_commands = []

    def _record_model_gate_command(command, args, *, program_name, **_kwargs):
        gate_commands.append((str(command), list(args), str(program_name)))
        return 23

    with (
        patch.object(research_app, "_print_workflow_status"),
        patch.object(
            research_app, "_load_reusable_continuous_forward_contract",
            return_value=(None, "synthetic missing"),
        ),
        patch.object(research_app, "_probe_continuous_fitted_candidates", return_value=None),
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
                "--filter-id", str(model_gate_settings.filter_id),
                "--model-architecture", str(model_gate_settings.model_architecture),
                "--experiment-profile", str(model_gate_settings.experiment_profile),
                "--seed", str(int(model_gate_settings.seed)),
                "--reuse-fitted-model",
            ],
            "apps/research.py model",
        )],
    )

    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
        patch("builtins.input", return_value=""),
        patch.object(research_app, "_run_continuous_forward_model_gate", return_value=29) as forward_gate,
        patch.object(research_app, "_run_continuous_rolling_mode_direct", return_value=31) as rolling_gate,
    ):
        menu_rc = research_app._interactive_model_research("apps/research.py model")
    check_true(
        "model_gate_only_continuous_profile_enter_routes_to_forward_gate_not_rolling_pit",
        menu_rc == 29
        and forward_gate.call_count == 1
        and rolling_gate.call_count == 0,
    )

    # [AI] A successful Forward run must redisplay the model menu; only the
    # explicit [0] exit returns to the parent Research menu.
    from contextlib import redirect_stdout
    from io import StringIO
    forward_menu_stdout = StringIO()
    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
        patch("builtins.input", side_effect=["", "0"]) as menu_input,
        patch.object(research_app, "_run_continuous_forward_model_gate", return_value=0) as forward_success,
        patch.object(research_app, "_run_continuous_rolling_mode_direct", return_value=31) as rolling_gate,
        redirect_stdout(forward_menu_stdout),
    ):
        forward_menu_rc = research_app._interactive_model_research("apps/research.py model")
    check_true(
        "continuous_forward_workflow_success_redisplays_model_menu_until_explicit_exit",
        forward_menu_rc == 0
        and forward_success.call_count == 1
        and rolling_gate.call_count == 0
        and menu_input.call_count == 2
        and forward_menu_stdout.getvalue().count("=== Continuous DL 模型研究與驗證 ===") == 2,
    )

    # Regression for B338: even if the current Training Model object carries a false
    # rolling flag (synthetic historical state), [4] belongs to the shared compare list
    # and must not be re-gated by that current Training Model.
    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
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

    # The seven displayed work types are a universal UI contract, not a
    # profile authorization list.  A false rolling flag must not hide anything.
    import contextlib
    import io
    universal_stdout = io.StringIO()
    with (
        patch.object(research_app, "get_breakout_quality_model_research_settings", return_value=model_gate_settings),
        patch("builtins.input", side_effect=["0"]),
        contextlib.redirect_stdout(universal_stdout),
    ):
        universal_rc = research_app._interactive_model_research("apps/research.py model")
    check_true(
        "formal_menu_seven_items_always_visible_even_when_training_rolling_flag_false",
        universal_rc == 0
        and all(f"[{item[0]}]" in universal_stdout.getvalue() and item[2] in universal_stdout.getvalue()
                for item in research_app._MODEL_RESEARCH_MENU_WORK_ITEMS)
        and "[BLOCKED]" not in universal_stdout.getvalue(),
    )

    app_source = (project_root / "services" / "research" / "breakout_quality_application.py").read_text(encoding="utf-8")
    import inspect
    report_source = inspect.getsource(inspect.getmodule(research_app._model_sop_view))
    # A stable work-type table may be reused by every configured research scope;
    # this test deliberately checks behavioral membership rather than the source
    # spelling of each render_menu_item call.
    stable_work_items = research_app._MODEL_RESEARCH_MENU_WORK_ITEMS
    check_true(
        "continuous_model_menu_uses_shared_forward_rolling_train_compare_robustness_workflow",
        tuple(choice for choice, _action, _label in stable_work_items) == tuple(str(i) for i in range(1, 8))
        and len({action for _choice, action, _label in stable_work_items}) == len(stable_work_items)
        and "查看目前Workflow、工件與模型報表" not in app_source
        and "Actual MFE×Safety Truth Geometry（只讀）" not in app_source,
    )
    # Scientific experiment registration must not control any stable model menu.
    # The standard workflow dispatch is owned by the formal Research application;
    # experiment-specific services remain isolated diagnostic implementations.
    from core.model_research_menu_contract import MODEL_RESEARCH_WORK_TYPES
    check_true(
        "research_menu_work_types_do_not_bind_algorithm_specific_producers",
        tuple(work.choice for work in MODEL_RESEARCH_WORK_TYPES) == tuple(str(i) for i in range(1, 8))
        and all(not hasattr(work, "producer_method") for work in MODEL_RESEARCH_WORK_TYPES)
        and "iter_exploratory_model_research_studies" not in app_source
        and "get_exploratory_model_research_study" not in app_source
        and "service_module" not in app_source[app_source.index("def _interactive_model_research"):app_source.index("def run_model_training_menu")],
    )
    daily_ranker_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    pit_score_source = (
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    fit_owner_source = (
        project_root / "services" / "breakout_quality" / "fitted_model_artifacts.py"
    ).read_text(encoding="utf-8")
    score_owner_source = (
        project_root / "filters" / "breakout_quality" / "ranking_score_store.py"
    ).read_text(encoding="utf-8")
    check_true(
        "model_workflow_ui_uses_mr_identity_one_status_table_and_inline_daily_target_progress",
        'print("Training Model：" + _model_identity_text(_model_display_id(settings.experiment_profile)))' in app_source
        and 'print(f"Training Profile：{settings.experiment_profile}")' not in app_source
        and 'def _model_identity_text(value: object, *, target: str = "console")' in report_source
        and 'paint(text, "cyan"' in report_source
        and 'def _print_model_action_status(' in app_source
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
        'return paint(str(label), "light_yellow", enabled=color, bold=True)' in report_source
        and 'return f"### {markdown_tone(label, \'light_yellow\', bold=True)}"' in report_source
        and 'f"seed={int(seed)} ({int(index)}/{int(total)})"' in app_source
        # Phase 7 pairs each source with its canonical per-seed dependency plan.
        # Verify strict alignment and the actual per-seed progress label,
        # rather than requiring the removed source-only loop spelling.
        and 'zip(seed_states, readiness.seeds, strict=True), start=1,' in app_source
        and 'seed_text = _seed_progress_text(seed, seed_index, len(seeds))' in app_source
        and '確認產生{mode_label}多模型Standard SOP比較；完整者REUSE，缺失者由canonical producer補建' not in app_source
        and '確認產生{mode_label} Robustness多模型Standard SOP比較；缺失seed由canonical producer補建' not in app_source
        and 'daily_data_progress = InlineProgress()' in pit_score_source
        and 'daily_data_progress.update(' in pit_score_source
        and 'daily_data_progress.finish()' in pit_score_source
        and 'render_training_unit_progress(' in pit_score_source
        and 'progress_callback=_fold_training_progress if compact_console else None' in pit_score_source
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
        and 'def _forward_checkpoint_import_issues(' in fit_owner_source
        and 'execution_plan=plan' in pit_score_source
        and 'checkpoint_sample_scope = str(' in fit_owner_source
        and 'checkpoint selected_epoch mismatch' not in pit_score_source,
    )

    # Test the Forward-to-Rolling FIT bridge with a registered, Rolling-authorized
    # profile.  The version owner rightly rejects fabricated profile identities;
    # synthetic output paths remain isolated by the fixture filter_id.
    rolling_settings = breakout_quality_policy.get_breakout_quality_workflow_settings(
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    rolling_settings = replace(rolling_settings, filter_id="synthetic_quality")
    forward_checkpoint_source = (Path("/synthetic/forward/model"), Path("/synthetic/forward/report.json"))
    with (
        tempfile.TemporaryDirectory() as rolling_test_dir,
        patch.object(
            research_app, "_rolling_publication_dir",
            return_value=Path(rolling_test_dir) / "unpublished_rolling",
        ),
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
    run_bound_checks(
        check_true,
        (
            ('forward_checkpoint_import_bridge_passes_model_and_report_as_one_atomic_argument_pair', synthetic_build_args[-4:] == ['--checkpoint-import-model-dir', str(forward_checkpoint_source[0].resolve()), '--checkpoint-import-report-path', str(forward_checkpoint_source[1].resolve())],),
            ('model_robustness_restart_reports_ready_seed_reuse_and_completed_seed_done', 'seed_text = _seed_progress_text(seed, seed_index, len(seeds))' in app_source and 'styled_workflow_status("[REUSE]")' in app_source and ('styled_workflow_status("[DONE]")' in app_source) and ('f" | {mode_label} Standard SOP READY"' in app_source),),
        ),
    )


    shared_training = breakout_quality_policy.get_breakout_quality_model_research_settings()
    shared_test = breakout_quality_policy.get_breakout_quality_model_test_settings().model_profiles
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
    from config import research_summary as summary_cfg
    from core.research_summary import get_research_summary_model_profiles

    check_true(
        "model_compare_test_membership_is_projected_from_summary",
        tuple(shared_test) == get_research_summary_model_profiles()
        and canonical_training_pair in shared_test
        and canonical_training_pair[1] == str(shared_training.experiment_profile),
    )
    # A new model joins every model workflow through the Summary roster once;
    # choosing a training model alone cannot create a second membership source.
    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE",
                      ("MR-SYNTHETIC-NEW", "synthetic_new_training_profile")):
        absent_training_rejected = raises_expected(
            ValueError, breakout_quality_policy.get_breakout_quality_model_test_settings,
        )
    check_true("training_selection_outside_summary_is_rejected", absent_training_rejected)

    original_test_pairs = tuple(shared_test)
    conflicting_model_pair = (original_test_pairs[0][0], "synthetic_different_profile")
    conflicting_profile_pair = ("MR-SYNTHETIC-CONFLICT", original_test_pairs[0][1])
    collision_rejections = []
    for conflicting_pair in (conflicting_model_pair, conflicting_profile_pair):
        with patch.object(summary_cfg, "RESEARCH_SUMMARY_ROWS", (
            *summary_cfg.RESEARCH_SUMMARY_ROWS,
            {"bindings": ({"rank_model": conflicting_pair},)},
        )):
            collision_rejections.extend((
                raises_expected(
                    ValueError,
                    breakout_quality_policy.get_breakout_quality_model_test_settings,
                ),
                raises_expected(
                    ValueError,
                    breakout_quality_policy.get_breakout_quality_model_workflow_profile_names,
                ),
                raises_expected(
                    ValueError,
                    lambda: breakout_quality_policy.is_breakout_quality_model_test_profile(
                        original_test_pairs[0][1]
                    ),
                ),
            ))
    check_true(
        "model_test_membership_conflicting_id_or_profile_is_rejected_by_canonical_policy",
        all(collision_rejections),
    )
    configured_workflow_profiles = breakout_quality_policy.get_breakout_quality_model_workflow_profile_names()
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
            breakout_quality_policy.get_breakout_quality_workflow_settings(
                experiment_profile=str(profile_name)
            ).rolling_authorized
            and breakout_quality_policy.get_breakout_quality_workflow_settings(
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
        and '_run_configured_model_comparison(program_name, rolling=True, confirm=True)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=False, confirm=True)' in app_source
        and '_run_configured_model_robustness(program_name, rolling=True, confirm=True)' in app_source,
    )
    check_true(
        "rolling_report_extension_refresh_reuses_pit_score_core_and_robustness_does_not_reenter_fit_when_only_audit_is_stale",
        "def _refresh_rolling_audit_only" in app_source
        and "確認只刷新Rolling audit/evidence（不重訓、不重算PIT scores）" in app_source
        and "def _rolling_robustness_score_core_ready" in app_source
        and "def load_selection_point_in_time_score_source_contract(" in score_owner_source
        and "def _rolling_robustness_score_core_contract" in app_source
        and "get_continuous_ranker_score_output_columns" in score_owner_source
        and "if core_ready:" in app_source
        and "if not inputs_prepared:" in app_source
        and '"audit-point-in-time-scores"' in app_source
        and "aggregate_robustness_row_extension" in app_source
        and "comparison_extension_aggregation" in report_source,
        note=(
            "Rolling comparison/robustness must treat model-specific evidence staleness as "
            "audit/report lifecycle only once canonical PIT score+manifest are valid"
        ),
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

    run_bound_checks(
        check,
        (
            ('event_scope_continuous_ranker_stays_out_of_binary_runtime_while_daily_runtime_export_is_separate', (True, True, True, True), ('SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES' in train_source, 'research-only continuous ranker artifact不得載入正式binary runtime contract' in artifact_source, 'SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES' in app_source, event_scope_runtime_export_rejected),),
            ('continuous_ranker_is_research_only_and_oos_follows_checkpoint_write', (True, True, True, True, True, True), ('"eligible": False' in ranker_source, 'OOS target transformation and model inference occur only after' in ranker_source, 'torch.save(' in ranker_source and ranker_source.index('torch.save(') < ranker_source.index('OOS target transformation and model inference occur only after'), 'score_frame["group_index"].duplicated().any()' in ranker_source, 'label_scope=profile.training_label_scope' in ranker_source, '"label_conditional": {}' in ranker_source),),
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

    # An additional research profile may recompose an existing training policy.
    # Inner epoch selection and final refit must not maintain separate copies of
    # the profile/recipe -> epoch-optimizer arguments or encoder initialization.
    from services.breakout_quality import train_continuous_ranker as ranker_stage_owner
    ranker_functions = {
        node.name: node
        for node in ast.parse(ranker_source).body
        if isinstance(node, ast.FunctionDef)
    }
    stage_call_errors = []
    for stage_name in ("select_epoch", "fit_final"):
        stage_calls = [
            node.func.id
            for node in ast.walk(ranker_functions[stage_name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        for owner_name in ("_initialize_ranker_fitting_stage", "_train_ranker_fitting_epoch"):
            if stage_calls.count(owner_name) != 1:
                stage_call_errors.append((stage_name, owner_name, stage_calls.count(owner_name)))
        for forbidden_name in ("_train_epoch", "pretrain_encoder_state", "build_grad_scaler"):
            if forbidden_name in stage_calls:
                stage_call_errors.append((stage_name, "duplicate_" + forbidden_name, stage_calls.count(forbidden_name)))
    for stage_name, expected_label, expected_ids in (
        ("select_epoch", "inner_train_epoch_selection", "train_ids"),
        ("fit_final", "full_selection_final_refit", "final_ids"),
    ):
        stage_nodes = [
            node for node in ast.walk(ranker_functions[stage_name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_initialize_ranker_fitting_stage"
        ]
        if len(stage_nodes) == 1:
            stage_node = stage_nodes[0]
            stage_kw = next((kw.value for kw in stage_node.keywords if kw.arg == "stage"), None)
            if not (
                isinstance(stage_kw, ast.Constant)
                and stage_kw.value == expected_label
                and len(stage_node.args) >= 4
                and isinstance(stage_node.args[3], ast.Name)
                and stage_node.args[3].id == expected_ids
            ):
                stage_call_errors.append((stage_name, "stage_or_fit_ids_mismatch"))
    common_epoch_calls = [
        node.func.id
        for node in ast.walk(ranker_functions["_train_ranker_fitting_epoch"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    check(
        "ranker_selection_and_refit_share_one_training_dispatch",
        ([], 1),
        (stage_call_errors, common_epoch_calls.count("_train_epoch")),
    )

    # Capability-driven call-contract test. All currently registered profiles,
    # including ones not used by the active UI, must inherit the SAME mapping.
    # No experiment identity, architecture list, or fixed config value is pinned.
    _previous_epoch_impl = ranker_stage_owner._train_epoch
    _epoch_calls = []
    _epoch_provider = object()
    def _capture_ranker_epoch(*_args, **_kwargs):
        _epoch_calls.append((_args, _kwargs))
        return 0.125
    try:
        ranker_stage_owner._train_epoch = _capture_ranker_epoch
        _stage_args = SimpleNamespace(
            seed=91, batch_size=13, gradient_clip_norm=1.5,
            train_prefetch_batches=2, train_prefetch_workers=3,
        )
        _bank = np.empty((2, 3, 1), dtype=np.float32)
        _context = np.empty((2, 0), dtype=np.float32)
        _ids = np.asarray([0, 1], dtype=np.int64)
        _dates = pd.Series(pd.to_datetime(["2020-01-02", "2020-01-02"]))
        _recipe_mismatches = []
        for _profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
            _candidate_profile = get_breakout_quality_experiment_profile(_profile_name)
            _candidate_recipe = get_continuous_ranker_execution_recipe(_profile_name)
            _epoch_calls.clear()
            _loss = ranker_stage_owner._train_ranker_fitting_epoch(
                None, object(), object(), _bank, _context, _ids, _ids, _dates,
                profile=_candidate_profile, execution_recipe=_candidate_recipe,
                args=_stage_args, plan=None, grad_scaler=None, epoch=3,
                raw_r_loss_name="sentinel_loss", raw_r_delta=0.75,
                adaptive_horizon_target_provider=_epoch_provider,
            )
            _observed = _epoch_calls[0][1] if len(_epoch_calls) == 1 else {}
            _expected = {
                "training_objective": _candidate_profile.training_objective,
                "batch_size": 13,
                "seed": 94,
                "gradient_clip_norm": 1.5,
                "plan": None,
                "grad_scaler": None,
                "prefetch_batches": 2,
                "prefetch_workers": 3,
                "pairwise_reduction": (
                    _candidate_recipe.pairwise_reduction
                    or continuous_ranker_runtime.CONTINUOUS_RANKER_PAIRWISE_REDUCTION_EQUAL_PAIR
                ),
                "pair_weight_policy": _candidate_recipe.objective_policy.pair_weight_policy,
                "secondary_pair_scope": _candidate_recipe.objective_policy.secondary_pair_scope,
                "secondary_pair_scope_threshold": _candidate_recipe.objective_policy.secondary_pair_scope_threshold,
                "raw_r_loss_name": "sentinel_loss",
                "raw_r_huber_delta_r": 0.75,
                "numerical_execution_policy": _candidate_profile.numerical_execution_policy,
                "bf16_backward_retry_scales": _candidate_profile.bf16_backward_retry_scales,
                "adaptive_horizon_target_provider": _epoch_provider,
            }
            if (
                _loss != 0.125
                or _observed != _expected
                or not _epoch_calls
                or _epoch_calls[0][0][5] is not _ids
                or _epoch_calls[0][0][6] is not _ids
                or _epoch_calls[0][0][7] is not _dates
            ):
                _recipe_mismatches.append(_profile_name)
        check(
            "ranker_epoch_dispatch_uses_canonical_recipe_for_registered_capabilities",
            [], _recipe_mismatches,
        )
    finally:
        ranker_stage_owner._train_epoch = _previous_epoch_impl

    # Both fitting stages retain the shared model/optimizer/scaler return contract.
    _stage_init_errors = []
    _model_sentinel = SimpleNamespace(eval=lambda: None)
    _optimizer_sentinel, _scaler_sentinel = object(), object()
    with patch.object(ranker_stage_owner, "_new_model_and_optimizer",
                      return_value=(_model_sentinel, _optimizer_sentinel)), patch.object(
        ranker_stage_owner, "build_grad_scaler", return_value=_scaler_sentinel,
    ):
        for _profile_name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
            _candidate_profile = get_breakout_quality_experiment_profile(_profile_name)
            for _stage in ("inner_train_epoch_selection", "full_selection_final_refit"):
                _initialized = ranker_stage_owner._initialize_ranker_fitting_stage(
                    None, np.empty((2, 3, 1), dtype=np.float32), np.empty((2, 0), dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int64),
                    pd.DataFrame({"date": pd.to_datetime(["2020-01-02"] * 2)}),
                    profile=_candidate_profile,
                    args=SimpleNamespace(seed=42, model_architecture="model_sentinel"),
                    plan=None, stage=_stage,
                )
                if _initialized != (_model_sentinel, _optimizer_sentinel, _scaler_sentinel, None):
                    _stage_init_errors.append((_profile_name, _stage))
    check("ranker_selection_refit_stage_preserves_shared_return_contract", [], _stage_init_errors)

    # Shared multi-head numerical guards and epoch selection consume the same
    # canonical training composition.  A new experiment which recomposes an
    # existing capability must not require another trainer-side handler list.
    import inspect as _ranker_inspect
    from services.breakout_quality.train_continuous_ranker import (
        _epoch_metric_improved as _compare_epoch_metrics,
        _guarded_retry_enabled as _guard_retry,
        _bf16_dynamic_backward_scaling_enabled as _bf16_guard,
    )
    from core.breakout_quality_registry import (
        NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING,
    )
    from core.breakout_quality_runtime import (
        get_continuous_ranker_training_policy,
        CONTINUOUS_RANKER_HEAD_LOSS_COMBINATION_EQUAL_MEAN_AVAILABLE as _available_heads,
        CONTINUOUS_RANKER_PAIRWISE_KIND_SCALAR as _scalar_pairwise,
    )
    _ranker_policies = {
        name: get_continuous_ranker_training_policy(name)
        for name in sorted({
            get_breakout_quality_experiment_profile(name).training_objective
            for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        })
    }
    _execution_plan_stub = SimpleNamespace(
        mixed_precision_enabled=True, autocast_dtype_name="bfloat16"
    )
    _model_stub = SimpleNamespace(
        same_batch_fp32_nonfinite_retry=True, recurrent_outer_autocast=True
    )
    _shared_errors = []
    for _objective, _policy in _ranker_policies.items():
        _expected_shared = bool(
            _policy.independent_head_composition is None
            and _policy.head_loss_combination == _available_heads
            and int(_policy.head_loss_component_count) == 2
            and _policy.pairwise_kind == _scalar_pairwise
        )
        _guard_allowed = False
        _bf16_allowed = False
        try:
            _guard_allowed = _guard_retry(
                _model_stub, _execution_plan_stub, None, _policy
            )
        except ValueError:
            pass
        try:
            _bf16_allowed = _bf16_guard(
                _model_stub, _execution_plan_stub, None, _policy,
                NUMERICAL_EXECUTION_POLICY_BF16_DYNAMIC_BACKWARD_SCALING,
            )
        except ValueError:
            pass
        if (_policy.uses_shared_scoped_head_loss != _expected_shared
                or _guard_allowed != _expected_shared
                or _bf16_allowed != _expected_shared):
            _shared_errors.append(_objective)
    check("registered_head_compositions_share_one_guard_and_loss_dispatch", [], _shared_errors)
    _epoch_source = _ranker_inspect.getsource(ranker_stage_owner._train_epoch)
    check_true(
        "trainer_uses_declared_shared_head_capability_not_second_handler_allowlist",
        "training_policy.uses_shared_scoped_head_loss" in _epoch_source
        and "_SHARED_SAFETY_SCOPED_MFE_LOSS_HANDLERS" not in _epoch_source,
    )

    # Assert decision boundaries independently: strict min_delta and exact
    # 1e-12 tie-break must be unchanged for all four metric families.
    _epoch_errors = []
    for _primary_dir, _secondary_dir in (("max", "min"), ("min", "max"), ("max", "max")):
        for _primary, _best, _delta in (
            (0.60, 0.50, 0.02), (0.40, 0.50, 0.02),
            (0.50, 0.50, 0.02), (0.5000000000005, 0.50, 0.02),
            (0.50, 0.50, 0.0), (0.52, 0.50, 0.02),
        ):
            for _secondary, _best_secondary in ((0.3, 0.2), (0.2, 0.3), (0.2, 0.2)):
                _primary_improved = (
                    _primary > _best + _delta
                    if _primary_dir == "max"
                    else _primary < _best - _delta
                )
                _secondary_improved = (
                    _secondary > _best_secondary
                    if _secondary_dir == "max"
                    else _secondary < _best_secondary
                )
                _expected_improved = _primary_improved or (
                    math.isclose(_primary, _best, rel_tol=0.0, abs_tol=1e-12)
                    and _secondary_improved
                )
                _actual_improved = _compare_epoch_metrics(
                    primary=_primary, best_primary=_best,
                    secondary=_secondary, best_secondary=_best_secondary,
                    min_delta=_delta, primary_direction=_primary_dir,
                    secondary_direction=_secondary_dir,
                )
                if _actual_improved != _expected_improved:
                    _epoch_errors.append((_primary_dir, _secondary_dir, _primary, _best, _delta))
    check("epoch_selection_comparator_strict_delta_and_ties_preserved", [], _epoch_errors)
    check_true(
        "epoch_selection_rejects_undeclared_comparator_direction",
        raises_expected(
            ValueError,
            lambda: _compare_epoch_metrics(
                primary=1.0, best_primary=0.0, secondary=0.0,
                best_secondary=0.0, min_delta=0.0,
                primary_direction="unknown", secondary_direction="max",
            ),
        ),
    )
    _allowed_rules = {
        "categorical_cross_entropy_then_event_brier",
        "regression_loss_then_spearman",
        "pareto_daily_then_global",
        "hmhs_pair_then_pr_auc",
        "daily_spearman_then_percentile_mse",
    }
    check(
        "registered_profiles_use_declared_epoch_selection_capabilities",
        [],
        [name for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
         if get_continuous_ranker_training_policy(
             get_breakout_quality_experiment_profile(name).training_objective
         ).epoch_selection_rule not in _allowed_rules],
    )
    # Independent oracle: categorical CE is the newly declared proper-score
    # family; historical precedence still follows RAW_R, DIRECT_HMHS, PARETO,
    # then daily rho. A pairwise-kind change cannot change the metric family.
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_CATEGORICAL_CE as _categorical_handler,
        CONTINUOUS_RANKER_LOSS_HANDLER_RAW_R as _raw_r_handler,
        CONTINUOUS_RANKER_TARGET_BUILDER_DIRECT_HMHS as _direct_hmhs_builder,
        CONTINUOUS_RANKER_TARGET_BUILDER_PARETO_COMPONENTS as _pareto_builder,
    )
    _epoch_family_errors = []
    for _name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES:
        _policy = get_continuous_ranker_training_policy(
            get_breakout_quality_experiment_profile(_name).training_objective
        )
        if _policy.loss_handler == _categorical_handler:
            _expected_family = "categorical_cross_entropy_then_event_brier"
        elif _policy.loss_handler == _raw_r_handler:
            _expected_family = "regression_loss_then_spearman"
        elif _policy.target_builder == _direct_hmhs_builder:
            _expected_family = "hmhs_pair_then_pr_auc"
        elif _policy.target_builder == _pareto_builder:
            _expected_family = "pareto_daily_then_global"
        else:
            _expected_family = "daily_spearman_then_percentile_mse"
        if _policy.epoch_selection_rule != _expected_family:
            _epoch_family_errors.append(_name)
    check("epoch_selection_family_matches_original_target_builder_precedence", [], _epoch_family_errors)
    _select_source = _ranker_inspect.getsource(ranker_stage_owner.select_epoch)
    check_true(
        "epoch_selector_calls_one_shared_numeric_comparator",
        _select_source.count("_epoch_metric_improved(") == 1
        and "training_policy.epoch_selection_rule" in _select_source,
    )

    summary["profiles"] = tuple(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES)
    return results, summary

def validate_breakout_quality_all_event_no_time_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_ALL_EVENT_NO_TIME_RANKER"
    results, summary, check, check_true = _start_case(case_id)

    from services.breakout_quality.train_continuous_ranker import (
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
    results, summary, check, check_true = _start_case(case_id)

    from filters.breakout_quality.models.factory import require_torch
    from services.breakout_quality.train_continuous_ranker import (
        PAIRWISE_TRAINING_CONTRACT,
        _daily_top_k_metrics,
        _date_coherent_batches,
        _pairwise_logistic_loss,
        _profile_contract,
        _training_semantics,
        _trade_alignment_metrics,
        parse_args as parse_continuous_ranker_args,
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

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    summary["training_objective"] = TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING
    return _finish_case(results, summary)


def _probe_continuous_oos_training_semantics(
    profile_name, *, mutate_manifest=None, mutate_report=None
):
    """Exercise the real OOS loader without a fitted checkpoint or score CSV."""

    from filters.breakout_quality import ranking_score_store as score_store
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe

    profile = get_breakout_quality_experiment_profile(profile_name)
    semantics = training_semantics(profile)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_path = root / "synthetic-model.pt"
        manifest_path = root / "manifest.json"
        report_path = root / score_store.CONTINUOUS_RANKER_REPORT_FILENAME
        model_path.write_bytes(b"synthetic-checkpoint")
        coverage = {
            "inference_eligible_groups": 1,
            "target_evaluable_groups": 1,
            "future_target_required_for_score": False,
        }
        eligibility = score_store.build_score_eligibility_contract(profile)
        manifest = {
            "filter_id": "synthetic",
            "model_architecture": "synthetic_architecture",
            "experiment_profile": profile.name,
            "training_objective": profile.training_objective,
            "training_label_scope": profile.training_label_scope,
            "training_sample_scope": profile.training_sample_scope,
            "training_semantics": json.loads(json.dumps(semantics)),
            "continuous_target_id": profile.continuous_target_id,
            "score_eligibility_contract": eligibility,
            "forward_score_coverage": coverage,
            "model": {},
            "outer_oos_policy": {
                "oos_start_date": "2021-01-01",
                "configured_oos_end_date": "2021-12-31",
            },
            "model_information_cutoff": "2020-12-31",
        }
        report = {
            "filter_id": "synthetic",
            "model_architecture": "synthetic_architecture",
            "experiment_profile": profile.name,
            "experiment_settings": profile.as_manifest_payload(),
            "training": {
                "objective": profile.training_objective,
                "loss": profile.loss_name,
                "sample_scope": profile.training_sample_scope,
                "training_label_scope": profile.training_label_scope,
                **json.loads(json.dumps(semantics)),
                "pairwise_reduction": recipe.pairwise_reduction,
                "pair_weight_policy": recipe.objective_policy.pair_weight_policy,
                "epoch_selection_metric": profile.epoch_selection_metric,
                "seed": 42,
            },
            "score_eligibility_contract": eligibility,
            "forward_score_coverage": coverage,
            "status": "RESULT_AVAILABLE",
        }
        if mutate_manifest is not None:
            mutate_manifest(manifest)
        if mutate_report is not None:
            mutate_report(report)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        score_store.load_continuous_ranker_oos_contract.cache_clear()
        with (
            patch.object(
                score_store, "resolve_filter_artifact_paths",
                return_value=SimpleNamespace(
                    manifest_path=manifest_path, model_path=model_path
                ),
            ),
            patch.object(score_store, "resolve_filter_model_output_dir", return_value=root),
            patch.object(
                score_store, "_validate_file_record_simple", return_value=None
            ),
        ):
            try:
                contract = score_store.load_continuous_ranker_oos_contract(
                    str(root), "synthetic", "synthetic_architecture", profile.name,
                    require_scores=False,
                )
            except ValueError as exc:
                return None, str(exc)
            return contract, ""


def validate_breakout_quality_daily_full_list_ndcg_pairwise_contract_case(_base_params):
    """Pin MR-13E as a full-list position-aware Delta-NDCG RankNet experiment."""

    case_id = "BREAKOUT_QUALITY_DAILY_FULL_LIST_NDCG_PAIRWISE"
    results, summary, check, check_true = _start_case(case_id)

    from filters.breakout_quality.models.factory import require_torch
    from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
    from services.breakout_quality.train_continuous_ranker import (
        parse_args as parse_continuous_ranker_args,
    )

    profile_e = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    spec_e = get_continuous_ranker_research_spec(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
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
        fake_score_table.to_csv(score_path, index=False, encoding="utf-8-sig", compression="gzip")
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
                # Isolated report-schema fixture does not materialize real CSV
                # inputs; current-source readiness is tested separately.
                patch.object(
                    ranking_store,
                    "_validate_current_forward_score_source_lineage",
                    return_value=None,
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
        wrong_weighting_rejected = raises_expected(ValueError, lambda: _load_contract())

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

    # SSOT: the loader must enforce the canonical training semantics for *every*
    # registered objective, not just a historical pairwise/listwise subset.
    from core.breakout_quality_registry import (
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    )
    from filters.breakout_quality.ranker_training_contract import (
        training_semantics as _canonical_training_semantics,
    )

    def _corrupt_one_nested_contract(payload):
        keys = [
            key for key, value in payload.items()
            if key.endswith("_contract") and isinstance(value, dict) and value
        ]
        specialized = next((key for key in keys if key != "pairwise_contract"), keys[0])
        first_key = next(iter(payload[specialized]))
        payload[specialized][first_key] = "altered-scientific-semantics"

    all_oos_contract_profiles = tuple(SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES)
    accepted_profiles = []
    for registered_name in all_oos_contract_profiles:
        actual_contract, error = _probe_continuous_oos_training_semantics(registered_name)
        if actual_contract is not None and not error:
            accepted_profiles.append(registered_name)
        else:
            summary.setdefault("generic_semantics_loader_errors", []).append(
                f"{registered_name}: {error}"
            )
    check(
        "all_registered_continuous_training_compositions_reuse_one_validated_oos_loader",
        all_oos_contract_profiles,
        tuple(accepted_profiles),
    )

    rejected_manifest_profiles = []
    rejected_report_profiles = []
    for registered_name in all_oos_contract_profiles:
        # Pick the actual active semantic capability for each profile; no
        # special-cased MR IDs or training-objective name list in the test.
        semantic_payload = _canonical_training_semantics(
            get_breakout_quality_experiment_profile(registered_name)
        )
        live_contract = next(
            (key for key, value in semantic_payload.items()
             if key.endswith("_contract") and value),
            "batching",
        )

        def _break_semantic(mapping, key=live_contract):
            if key == "batching":
                mapping[key] = "wrong_date_batching"
            else:
                first_key = next(iter(mapping[key]))
                mapping[key][first_key] = "wrong_training_semantics"

        _loaded, error = _probe_continuous_oos_training_semantics(
            registered_name,
            mutate_manifest=lambda data: _break_semantic(data["training_semantics"]),
        )
        if _loaded is None and "training_semantics" in error:
            rejected_manifest_profiles.append(registered_name)
        _loaded, error = _probe_continuous_oos_training_semantics(
            registered_name,
            mutate_report=lambda data: _break_semantic(data["training"]),
        )
        if _loaded is None and "report.training" in error:
            rejected_report_profiles.append(registered_name)
    check(
        "every_registered_objective_rejects_tampered_manifest_semantics",
        all_oos_contract_profiles, tuple(rejected_manifest_profiles),
    )
    check(
        "every_registered_objective_rejects_tampered_report_semantics",
        all_oos_contract_profiles, tuple(rejected_report_profiles),
    )

    ao_name = DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    for label, name in (("ao_shared_hs", ao_name),):
        canonical = _canonical_training_semantics(get_breakout_quality_experiment_profile(name))
        semantic_key = next(
            key for key, value in canonical.items()
            if key.endswith("_contract") and key != "pairwise_contract" and value
        )
        contract, error = _probe_continuous_oos_training_semantics(
            name,
            mutate_manifest=lambda data, key=semantic_key: data["training_semantics"].pop(key),
        )
        check_true(
            f"{label}_missing_required_head_contract_fails_closed",
            contract is None and semantic_key in error,
        )
        contract, error = _probe_continuous_oos_training_semantics(
            name,
            mutate_manifest=lambda data, key=semantic_key: _corrupt_one_nested_contract(
                {key: data["training_semantics"][key]}
            ),
        )
        check_true(
            f"{label}_tampered_head_semantics_fails_closed",
            contract is None and semantic_key in error,
        )
        contract, error = _probe_continuous_oos_training_semantics(
            name,
            mutate_report=lambda data, key=semantic_key: _corrupt_one_nested_contract(
                {key: data["training"][key]}
            ),
        )
        check_true(
            f"{label}_report_head_semantics_fails_closed",
            contract is None and semantic_key in error,
        )

    _contract, error = _probe_continuous_oos_training_semantics(
        ao_name,
        mutate_manifest=lambda data: data.pop("training_semantics"),
    )
    check_true(
        "modern_shared_hs_missing_entire_training_semantics_cannot_reuse",
        _contract is None and "training_semantics" in error,
    )
    _contract, error = _probe_continuous_oos_training_semantics(
        ao_name,
        mutate_manifest=lambda data: data["training_semantics"].__setitem__(
            "batching", "shuffled_unique_group_batches"
        ),
    )
    check_true(
        "modern_shared_hs_batching_mismatch_cannot_reuse",
        _contract is None and "batching" in error,
    )
    _contract, error = _probe_continuous_oos_training_semantics(
        ao_name,
        mutate_report=lambda data: data["training"].__setitem__(
            "unrecognized_future_contract", {"unsafe": True}
        ),
    )
    check_true(
        "unknown_semantics_contract_field_in_report_is_rejected",
        _contract is None and "unexpected_keys" in error,
    )
    _contract, error = _probe_continuous_oos_training_semantics(
        ao_name,
        mutate_manifest=lambda data: data["training_semantics"].pop(
            "raw_r_regression_contract", None
        ),
    )
    check_true(
        "historical_later_added_null_slot_may_be_absent_for_newer_composition",
        _contract is not None and not error,
    )
    for label, field, invalid in (
        ("reduction", "pairwise_reduction", "invalid_reduction"),
        ("pair_policy", "pair_weight_policy", "invalid_policy"),
        ("epoch_selection", "epoch_selection_metric", "invalid_metric"),
    ):
        _contract, error = _probe_continuous_oos_training_semantics(
            ao_name,
            mutate_report=lambda data, key=field, value=invalid: data["training"].__setitem__(
                key, value
            ),
        )
        check_true(
            f"published_{label}_must_match_canonical_training_recipe",
            _contract is None and field in error,
        )

    _contract, error = _probe_continuous_oos_training_semantics(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        mutate_manifest=lambda data: data.pop("training_semantics"),
    )
    check_true(
        "newer_full_list_pairwise_missing_manifest_semantics_cannot_reuse",
        _contract is None and "training_semantics" in error,
    )

    summary["profile"] = DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    summary["model_research_id"] = spec_e.model_research_id
    return results, summary




def validate_breakout_quality_full_horizon_target_components_contract_case(_base_params):
    """Protect reusable full-horizon opportunity and pure-MFE target mathematics."""
    case_id = "BREAKOUT_QUALITY_FULL_HORIZON_TARGET_COMPONENTS"
    results, summary, check, check_true = _start_case(case_id)

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
    from types import SimpleNamespace
    from filters.breakout_quality.daily_ranker_data import build_daily_ranker_split
    from filters.breakout_quality.ranker_sample_contract import resolve_forward_oos_score_group_ids
    split_dates = pd.to_datetime(["2019-06-03"] * 24 + ["2020-12-01"] * 24 + ["2021-06-01"] * 24)
    target_valid = np.ones(len(split_dates), dtype=bool)
    # One missing target and one unavailable prediction-time context in each period.
    target_valid[[0, 1, 24, 25, 48, 49]] = False
    split_bundle = SimpleNamespace(
        group_table=pd.DataFrame({"date": split_dates, "label_eval_end_date": split_dates + pd.Timedelta(days=1)}),
        target_valid=target_valid,
        summary={"training_universe_start_date": "2019-01-01"},
        outer_policy={"selection_end_date": "2020-12-31", "oos_start_date": "2021-01-01", "effective_oos_end_date": "2021-12-31"},
    )
    split = build_daily_ranker_split(split_bundle, inner_validation_months=1)
    check(
        "target_invalid_or_context_unavailable_rows_excluded_from_every_supervised_split",
        ([*range(2, 24)], [*range(26, 48)], [*range(2, 24), *range(26, 48)], [*range(50, 72)]),
        (split.inner_train_ids.tolist(), split.validation_ids.tolist(), split.selection_ids.tolist(), split.oos_ids.tolist()),
    )
    check(
        "target_invalid_forward_rows_remain_in_feature_eligible_score_universe",
        list(range(48, 72)), resolve_forward_oos_score_group_ids(split_bundle).tolist(),
    )
    return _finish_case(results, summary)

def _check_centered_log_volume_input(check_true):
    """AI: PIT, schema and value contracts for the surviving 300x10 representation."""
    from filters.breakout_quality.daily_ranker_data import LazyDailyFeatureBank
    from filters.breakout_quality.features import centered_log_volume_windows
    from filters.breakout_quality.models.architectures import ARCHITECTURE_DESCRIPTORS
    from filters.breakout_quality.models.spec import get_model_spec

    length = int(DEFAULT_LABEL_POLICY.feature_window_bars)
    count = length + 20
    index = pd.bdate_range("2017-01-02", periods=count)
    t = np.arange(count, dtype=np.float64)

    def frame(scale, volume):
        close = scale * (1.0 + t / count)
        return pd.DataFrame({
            "Open": close * .99, "High": close * 1.02,
            "Low": close * .98, "Close": close, "Volume": volume,
        }, index=index)

    stock = frame(40.0, np.where(t % 17 == 0, 0.0, np.expm1(3.0 + t / 40.0)))
    illiquid = frame(12.0, np.where(t % 71 == 0, 1000.0, 0.0))
    benchmark = frame(80.0, np.expm1(5.0 + np.sin(t / 13.0)))
    positions = np.asarray([length - 1, length + 3, length + 8, length + 11])
    benchmark_positions = positions + 1
    tickers = np.asarray([0, 1, 0, 1])

    def bank(frames=(stock, illiquid), market=benchmark, *, enabled=True, **options):
        return LazyDailyFeatureBank(
            frames=frames, benchmark=market, ticker_ids=tickers,
            source_positions=positions, benchmark_positions=benchmark_positions,
            policy=DEFAULT_LABEL_POLICY, centered_log_volume=enabled, **options,
        )

    base = bank(enabled=False)
    replaced = bank()
    order = np.asarray([3, 0, 2, 0, 1])
    base_values = base[order]
    values = replaced[order]
    check_true("volume_replacement_keeps_original_ten_channel_width",
               values.shape == base_values.shape == (len(order), length, 10)
               and values.dtype == np.float32)
    check_true("volume_replacement_preserves_all_eight_price_channels_bitwise",
               np.array_equal(values[:, :, :4], base_values[:, :, :4])
               and np.array_equal(values[:, :, 5:9], base_values[:, :, 5:9]))
    check_true("volume_replacement_manifest_has_distinct_ten_channel_semantics",
               replaced.feature_columns == replaced.manifest_feature_columns
               and len(replaced.feature_columns) == 10
               and replaced.feature_columns[4] == "Stock_Centered_Log_Volume"
               and replaced.feature_columns[9] == "Benchmark_Centered_Log_Volume")
    check_true("volume_replacement_preserves_nonstandard_indexing",
               np.array_equal(replaced[-1], replaced[3])
               and np.array_equal(replaced[1:3], replaced[np.asarray([1, 2])])
               and np.array_equal(replaced[np.asarray([True, False, True, False])], replaced[[0, 2]])
               and replaced[[]].shape == (0, length, 10))

    for out, group in enumerate(order):
        for column, source, position in (
            (4, (stock, illiquid)[tickers[group]], positions[group]),
            (9, benchmark, benchmark_positions[group]),
        ):
            raw_volume = source["Volume"].to_numpy()[position - length + 1:position + 1]
            logs = np.log1p(np.maximum(raw_volume, 0.0))
            expected = (logs - np.median(logs)).astype(np.float32)
            check_true(f"centered_volume_matches_independent_window_formula_{out}_{column}",
                       np.array_equal(values[out, :, column], expected))
    check_true("centered_volume_preserves_amplitude_removed_by_iqr_normalization",
               not np.allclose(values[1, :, 4], base_values[1, :, 4]))

    constants = np.stack([np.zeros(length), np.full(length, 7.0), np.full(length, -2.0)])
    check_true("centered_volume_zero_constant_and_clamped_volume_are_finite_zero",
               np.array_equal(centered_log_volume_windows(constants), np.zeros_like(constants)))
    for bad in (np.nan, np.inf, -np.inf):
        missing = constants.copy()
        missing[0, 0] = bad
        check_true(f"centered_volume_rejects_missing_observation_{bad}",
                   raises_expected(ValueError, lambda: centered_log_volume_windows(missing)))
    for invalid in (np.ones(length), np.empty((2, 0))):
        check_true(f"centered_volume_rejects_invalid_window_shape_{invalid.shape}",
                   raises_expected(ValueError, lambda: centered_log_volume_windows(invalid)))

    future_stock, future_illiquid, future_benchmark = (item.copy() for item in (stock, illiquid, benchmark))
    for future in (future_stock, future_illiquid, future_benchmark):
        future.iloc[length + 14:, :] *= 1000.0
    check_true("volume_replacement_cannot_read_later_stock_or_benchmark_bars",
               np.array_equal(values, bank((future_stock, future_illiquid), future_benchmark)[order]))
    scaled = [item.copy() for item in (stock, illiquid, benchmark)]
    for item, factor in zip(scaled, (2.0, 4.0, 8.0)):
        item.loc[:, ["Open", "High", "Low", "Close"]] *= factor
    check_true("volume_replacement_is_invariant_to_retrospective_price_rescaling",
               np.array_equal(values, bank(tuple(scaled[:2]), scaled[2])[order]))
    check_true("volume_replacement_rejects_unapproved_alternate_price_composition",
               raises_expected(ValueError, lambda: bank(feature_frames=(stock, illiquid))))
    check_true("centered_volume_architectures_declare_the_same_spec_input_path",
               all("centered_log_volume" in get_model_spec(item.architecture_id).sequence_input_paths
                   for item in ARCHITECTURE_DESCRIPTORS.values()
                   if item.has_capability("centered_log_volume")))


def validate_breakout_quality_reusable_model_component_contract_case(_base_params):
    """Protect reusable ranker components without pinning closed experiment identities."""

    case_id = "BREAKOUT_QUALITY_REUSABLE_MODEL_COMPONENTS"
    results, summary, check, check_true = _start_case(case_id)
    _check_centered_log_volume_input(check_true)

    from core.breakout_quality_registry import (
        SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_research_spec,
    )
    from core.breakout_quality_policy import (
        get_continuous_ranker_execution_recipe,
    )
    from core.strategy_compare_policy import (
        get_strategy_comparison_settings,
        get_strategy_rolling_test_modes,
    )
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_required_artifact_sources,
    )
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
        INCEPTION_TIME_V1,
        get_model_spec,
        model_spec_from_manifest,
    )
    from filters.breakout_quality.models.spec_registry import MODEL_SPEC_BUILDERS
    from filters.breakout_quality.ranker_training_contract import training_semantics

    torch, _nn = require_torch()

    # Execution recipe is derived from canonical profile/spec but intentionally omits
    # MR identity.  Historical per-profile flags remain frozen evidence; current model
    # workflow authorization is instead derived from the configured Training Profile /
    # Model Compare-Test List SSOT and must not require a second profile authorization list.
    current_dependency_profiles = set()
    for mode in get_strategy_rolling_test_modes():
        mode_settings = get_strategy_comparison_settings(str(mode["profile_id"]))
        _required_params, required_dl_sources, _runtime_dl_sources = (
            resolve_required_artifact_sources(mode_settings)
        )
        current_dependency_profiles.update(
            str(mode_settings.dl_sources[dl_id].experiment_profile)
            for dl_id in required_dl_sources
            if str(mode_settings.dl_sources[dl_id].experiment_profile)
            in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
        )

    check_true(
        "current_compare_dependencies_use_shared_model_workflow_authorization",
        all(
            breakout_quality_policy.get_breakout_quality_workflow_settings(
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
    run_bound_checks(
        check,
        (
            ('current_time_validation_never_bypasses_historical_pit_authorization', [], authorization_inversions,),
            ('active_and_legacy_model_architecture_sets_are_disjoint_and_exhaustive', (set(), set(SUPPORTED_MODEL_ARCHITECTURES)), (set(ACTIVE_MODEL_ARCHITECTURES).intersection(LEGACY_MODEL_ARCHITECTURES), set(ACTIVE_MODEL_ARCHITECTURES).union(LEGACY_MODEL_ARCHITECTURES)),),
            ('model_spec_registry_is_exhaustive_without_second_architecture_matrix', set(SUPPORTED_MODEL_ARCHITECTURES), set(MODEL_SPEC_BUILDERS),),
        ),
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
    run_bound_checks(
        check_true,
        (
            ('architecture_descriptor_runtime_builder_keys_are_registered', {descriptor.runtime_builder_key for descriptor in ARCHITECTURE_DESCRIPTORS.values()} <= set(registered_runtime_builder_keys()),),
            ('active_descriptors_have_unique_explicit_active_order', all((get_architecture_descriptor(name).active_order is not None for name in ACTIVE_MODEL_ARCHITECTURES)) and len({get_architecture_descriptor(name).active_order for name in ACTIVE_MODEL_ARCHITECTURES}) == len(ACTIVE_MODEL_ARCHITECTURES),),
        ),
    )

    recurrent_fp32_unidirectional = []
    recurrent_fp32_bidirectional = []
    recurrent_fp32_bidirectional_dual = []

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

    # Generic independent-head composition is a capability, not an MR-specific matrix.
    # Derive coverage from all registered profiles and verify the same owner drives
    # target column order, forward routing and the canonical loss component count.
    from core.breakout_quality_runtime import (
        ContinuousRankerIndependentHeadComposition,
        ContinuousRankerPairwiseHead,
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        HEAD_TARGET_WITH_DETACHED_PRIMARY_SAFETY_RANK,
        get_continuous_ranker_training_policy,
    )
    from services.breakout_quality.train_continuous_ranker import (
        _combine_pairwise_head_supervision,
        _compute_independent_head_pairwise_loss,
        _same_date_average_rank_percentile,
    )
    registered_objectives = sorted({
        get_breakout_quality_experiment_profile(name).training_objective
        for name in SUPPORTED_CONTINUOUS_RANKER_RESEARCH_PROFILES
    })
    independent_policies = [
        (name, get_continuous_ranker_training_policy(name))
        for name in registered_objectives
        if get_continuous_ranker_training_policy(name).independent_head_composition is not None
    ]
    check_true("registered_independent_head_capabilities_are_discovered_from_runtime", bool(independent_policies))
    for objective, policy in independent_policies:
        comp = policy.independent_head_composition
        check_true(
            f"independent_head_contract_{objective}",
            len(comp.heads) == int(policy.head_loss_component_count)
            and tuple(head.target_column for head in comp.heads) == tuple(range(len(comp.heads)))
            and bool(comp.forward_method)
            and bool(comp.target_error)
            and bool(comp.architecture_error),
        )
    check_true(
        "independent_head_composition_rejects_unaligned_target_column_order",
        raises_expected(
            ValueError,
            lambda: ContinuousRankerIndependentHeadComposition(
                "forward_example_heads",
                (ContinuousRankerPairwiseHead(0), ContinuousRankerPairwiseHead(0)),
                "bad target",
                "bad architecture",
            ),
        ),
    )
    check_true(
        "independent_head_composition_rejects_predicted_safety_context_without_safety_pair_weight",
        raises_expected(
            ValueError,
            lambda: ContinuousRankerPairwiseHead(
                1,
                CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
                HEAD_TARGET_WITH_DETACHED_PRIMARY_SAFETY_RANK,
            ),
        ),
    )
    weighted_policy = next(
        (policy for _objective, policy in independent_policies
         if any(head.target_transform == HEAD_TARGET_WITH_DETACHED_PRIMARY_SAFETY_RANK
                for head in policy.independent_head_composition.heads)),
        None,
    )
    check_true("detached_safety_rank_context_has_registered_generic_head_capability", weighted_policy is not None)
    if weighted_policy is not None:
        class _DeterministicDuoModel:
            def __init__(self, primary, secondary):
                self.primary = primary
                self.secondary = secondary

            def forward_safety_mfe_heads(self, _x, _context):
                return self.primary, self.secondary

        first = torch.tensor(
            [[-.3, .1], [-.7, .9], [.8, -.1], [-.2, .5]],
            dtype=torch.float32, requires_grad=True,
        )
        second = torch.tensor(
            [[-.1, .2], [-.2, .7], [.6, -.4], [-.8, 1.2]],
            dtype=torch.float32, requires_grad=True,
        )
        target = torch.tensor(
            [[.1, .2], [.9, .7], [.3, .8], [.7, .4]], dtype=torch.float32,
        )
        dates = np.asarray(["2026-01-02"] * 4)
        model = _DeterministicDuoModel(first, second)
        actual_loss, actual_count = _compute_independent_head_pairwise_loss(
            torch, model, None, None, target, dates,
            training_policy=weighted_policy,
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
        )
        detached_rank = _same_date_average_rank_percentile(
            torch, torch.softmax(first.float(), dim=1)[:, LABEL_PASS].detach(), dates
        )
        expected_loss, expected_count = _combine_pairwise_head_supervision(
            torch,
            (first, second),
            (target[:, 0], torch.stack([target[:, 1], detached_rank], dim=1)),
            dates,
            training_policy=weighted_policy,
            pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
            pair_weight_policies=(
                CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
                CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
            ),
        )
        actual_grads = torch.autograd.grad(actual_loss, (first, second), retain_graph=True)
        expected_grads = torch.autograd.grad(expected_loss, (first, second))
        check_true(
            "independent_head_detached_safety_rank_preserves_exact_loss_and_gradients",
            actual_count == expected_count
            and torch.equal(actual_loss, expected_loss)
            and all(torch.equal(left, right) for left, right in zip(actual_grads, expected_grads)),
        )
        check_true(
            "independent_head_forward_method_missing_fails_fast",
            raises_expected(
                ValueError,
                lambda: _compute_independent_head_pairwise_loss(
                    torch, object(), None, None, target, dates,
                    training_policy=weighted_policy,
                    pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
                    pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
                ),
            ),
        )
        check_true(
            "independent_head_target_width_mismatch_fails_fast",
            raises_expected(
                ValueError,
                lambda: _compute_independent_head_pairwise_loss(
                    torch, model, None, None, target[:, :1], dates,
                    training_policy=weighted_policy,
                    pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
                    pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
                ),
            ),
        )
        class _IncorrectHeadCountModel:
            def forward_safety_mfe_heads(self, _x, _context):
                return (first,)

        check_true(
            "independent_head_output_count_mismatch_fails_fast",
            raises_expected(
                ValueError,
                lambda: _compute_independent_head_pairwise_loss(
                    torch, _IncorrectHeadCountModel(), None, None, target, dates,
                    training_policy=weighted_policy,
                    pairwise_reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
                    pair_weight_policy=CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_NONE,
                ),
            ),
        )
        from dataclasses import replace
        check_true(
            "independent_head_policy_loss_count_mismatch_fails_fast",
            raises_expected(
                ValueError,
                lambda: replace(weighted_policy, head_loss_component_count=3),
            ),
        )

    summary["components"] = ("execution_recipe", "architecture_registry", "independent_head_composition")
    from filters.breakout_quality.inference import strict_parallel_batched_logits
    torch, _nn = require_torch()
    torch.manual_seed(42)
    probe_features = np.random.default_rng(42).normal(size=(5, 24, 10)).astype(np.float32)
    probe_context = np.empty((5, 0), dtype=np.float32)
    order = np.asarray([4, 1, 3, 0], dtype=np.int64)
    for architecture in ACTIVE_MODEL_ARCHITECTURES:
        spec = get_model_spec(architecture)
        feature_count = probe_features.shape[-1]
        sequence_length = spec.input_window_bars or 24
        probe_features = np.random.default_rng(42).normal(size=(5, sequence_length, feature_count)).astype(np.float32)
        model = build_active_model(feature_count, 0, architecture=architecture).eval()
        shared_heads = get_architecture_descriptor(architecture).has_capability("shared_safety_mfe")
        heads = (None, "raw_safety", "conditional_mfe", "both") if shared_heads else (None, "primary")
        with torch.no_grad():
            x = torch.from_numpy(probe_features[order])
            context = torch.from_numpy(probe_context[order])
            for head in heads:
                expected = model(x, context) if head is None else model.forward_output_head(x, context, head)
                observed = strict_parallel_batched_logits(
                    torch, model, probe_features, probe_context, indices=order,
                    batch_size=2, workers=2, execution_plan=None, output_head=head,
                )
                check_true(
                    f"current_batched_inference_preserves_width_head_and_row_order_{architecture}_{head}",
                    observed.shape == tuple(expected.shape)
                    and np.allclose(observed, expected.numpy(), rtol=1e-5, atol=1e-6),
                )
            if shared_heads:
                safety, mfe = model.forward_safety_mfe_heads(x, context)
                check_true(
                    "current_shared_head_output_order_is_safety_then_mfe",
                    torch.equal(model.forward_output_head(x, context, "both"), torch.cat((safety, mfe), dim=1)),
                )
    return _finish_case(results, summary)



def validate_breakout_quality_low_adverse_target_component_contract_case(_base_params):
    """Protect the reusable full-horizon negative-adverse target transform."""
    case_id = "BREAKOUT_QUALITY_LOW_ADVERSE_TARGET_COMPONENT"
    results, summary, check, check_true = _start_case(case_id)

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
    return _finish_case(results, summary)







def validate_breakout_quality_shared_ah_contract_case(_base_params):
    """Protect the retained Shared-AH model and generic detached-head learning contract."""

    case_id = "BREAKOUT_QUALITY_MR13AK_SHARED_AH"
    results, summary, check, check_true = _start_case(case_id)

    import torch
    from config.breakout_quality import (
        BREAKOUT_QUALITY_MODEL_TEST_PROFILES,
    )
    from core.breakout_quality_registry import (
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_research_spec,
    )
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_WEIGHTED_MFE_PAIRWISE_RANKING,
    )
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_WEIGHTED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_PAIR_WEIGHT_POLICY_PRODUCT_PREDICTED_SAFETY,
        CONTINUOUS_RANKER_TARGET_BUILDER_SAFETY_RAW_MFE,
        get_continuous_ranker_pair_weight_policy,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
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

    check_true(
        "shared_ah_final_mfe_topology_semantics_are_owned_by_model_spec",
        model_spec.final_mfe_topology_contract()
        == {
            "architecture": "shared_encoder_independent_raw_safety_and_raw_mfe_heads",
            "mfe_head_inputs": "shared_latent_only_no_safety_prediction_input",
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


    comparison_pairs = [
        (str(model_id), str(profile_name))
        for model_id, profile_name in BREAKOUT_QUALITY_MODEL_TEST_PROFILES
    ]
    current_training_pair = tuple(
        str(value) for value in breakout_quality_config.BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
    )
    check_true(
        "continuous_dl_comparison_membership_uses_current_summary_and_keeps_ak_as_historical",
        current_training_pair in comparison_pairs
        and "MR-13AK" not in {model_id for model_id, _profile_name in comparison_pairs}
        and "MR-13AH" not in {model_id for model_id, _profile_name in comparison_pairs}
        and get_continuous_ranker_research_spec(profile.name).model_research_id == "MR-13AK"
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

    return _finish_case(results, summary)


































def validate_breakout_quality_true_hs_scoped_pair_membership_contract_case(_base_params):
    """Protect true-HS Conditional-MFE list membership and lexicographic attribution semantics."""

    case_id = "BREAKOUT_QUALITY_TRUE_HS_SCOPED_PAIR_MEMBERSHIP"
    results, summary, check, check_true = _start_case(case_id)

    import torch
    from core.breakout_quality_registry import (
        DAILY_UNIVERSAL_SHARED_SAFETY_HS_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_continuous_ranker_research_spec,
    )
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        TRAINING_OBJECTIVE_DAILY_SHARED_SAFETY_HS_CONDITIONAL_MFE_PAIRWISE_RANKING,
    )
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_LOSS_HANDLER_SHARED_SAFETY_SCOPED_MFE_DUO_PAIRWISE,
        CONTINUOUS_RANKER_SECONDARY_PAIR_SCOPE_PRIMARY_TARGET_MIN,
        CONTINUOUS_RANKER_TARGET_BUILDER_HS_CONDITIONAL_MFE,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
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
    run_bound_checks(
        check_true,
        (
            ('true_hs_scope_has_same_gate_ak_attribution_reference_only', research.model_gate_reference_profile_name == DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE and research.reference_profile_name is None and (research.evaluation_reference_profile_name is None),),
            ('true_hs_historical_forward_gate_follows_current_membership_ssot', _historical_forward_gate_matches_current_membership(profile, research),),
        ),
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

    group_table = _five_item_hs_group_table(include_label=False)
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

    return _finish_case(results, summary)










def validate_breakout_quality_fitted_model_lifecycle_contract_case(_base_params):
    from core.breakout_quality_registry import DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE
    """Historical regression: report refresh must never erase a compatible fitted model."""

    case_id = "BREAKOUT_QUALITY_FITTED_MODEL_LIFECYCLE"
    results, summary, check, check_true = _start_case(case_id)

    from services.breakout_quality.fitted_model_artifacts import (
        FittedModelConflictError,
        load_reusable_fitted_model_contract,
        write_fitted_model_contract,
    )
    from services.breakout_quality import rolling_model_source as rolling_source_owner
    from services.breakout_quality import forward_model_source as forward_source_owner

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        model_path.write_bytes(b"synthetic-fitted-checkpoint")
        identity = {
            "filter_id": "breakout_quality_v1",
            "model_architecture": "synthetic_architecture",
            "experiment_profile": "synthetic_profile",
            "seed": 42,
            "training_settings": {"learning_rate": 0.0003},
        }
        sidecar_path = write_fitted_model_contract(
            model_dir=model_dir,
            fitting_identity=identity,
            selected_epoch=3,
            epoch_selection={"best_epoch": 3},
            final_refit_history=[{"epoch": 1}],
            trainable_parameter_count=10,
            total_parameter_count=12,
        )
        contract, reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity=identity,
            final_manifest_path=root / "missing_manifest.json",
            report_path=root / "missing_report.json",
        )
        check_true(
            "fitted_model_sidecar_is_reusable_without_forward_report_or_final_manifest",
            contract is not None
            and reason is None
            and contract.source == "fitted_model_manifest"
            and contract.selected_epoch == 3
            and sidecar_path.is_file(),
        )

        changed_identity = dict(identity)
        changed_identity["training_settings"] = {"learning_rate": 0.0004}
        changed_contract, changed_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity=changed_identity,
            final_manifest_path=root / "missing_manifest.json",
            report_path=root / "missing_report.json",
        )
        check_true(
            "fitted_model_reuse_requires_exact_fitting_identity",
            changed_contract is None and "fitting identity mismatch" in str(changed_reason),
        )

        model_path.write_bytes(b"synthetic-conflicting-checkpoint")
        conflict_raised = raises_expected(FittedModelConflictError, lambda: load_reusable_fitted_model_contract(
                model_dir=model_dir,
                expected_identity=identity,
                final_manifest_path=root / "missing_manifest.json",
                report_path=root / "missing_report.json",
            ))
        check_true(
            "same_fitting_identity_with_different_checkpoint_bytes_fails_fast",
            conflict_raised,
        )

    # Data-lineage regression: changed input bytes must not reuse a model just
    # because train sample counts, input shape, and optimizer settings match.
    from services.breakout_quality.fitted_model_artifacts import (
        forward_fitted_source_provenance_issues,
        source_training_identity,
    )
    from filters.breakout_quality import source_inventory
    from filters.breakout_quality.model_source_provenance import (
        pit_published_source_provenance_issues,
    )
    from filters.breakout_quality import ranking_score_store

    # September 12 input-contract migration: identical metadata alone must
    # NEVER approve old PIT bytes, but a CSV-only V1 fingerprint plus a full
    # source scan with no non-finite values or duplicate dates can prove that
    # both sanitizers produced the same input. The old manifests stay intact.
    with tempfile.TemporaryDirectory() as temp_dir:
        test_root = Path(temp_dir)
        csv_dir = test_root / "historical_csv"
        csv_dir.mkdir()
        csv_file = csv_dir / "2330.csv"
        clean_rows = [
            {"Date": "2020-01-02", "Open": 10, "High": 12, "Low": 9, "Close": 11, "Volume": 100},
            {"Date": "2020-01-03", "Open": 11, "High": 13, "Low": 10, "Close": 12, "Volume": 101},
        ]
        pd.DataFrame(clean_rows).to_csv(csv_file, index=False)
        with patch.object(source_inventory, "get_dataset_dir", return_value=str(csv_dir)):
            historical_inventory = source_inventory._build_source_data_inventory(
                test_root, "reduced", legacy_csv_only=True,
            )
            current_inventory = source_inventory.build_source_data_inventory(test_root, "reduced")
            source_records = {"features": {"sha256": "unchanged-feature-bytes"}}
            target_records_v1 = {"target": {"sha256": "unchanged-target-bytes"}}
            source_dataset = {
                "dataset": "reduced", "policy": {"window": 300},
                "source_data_inventory": historical_inventory,
                "source_dataset_artifacts": source_records,
            }
            source_target = {"artifacts": target_records_v1, "target_contract": {"target": "safety"}}
            current_dataset = {
                **source_dataset, "source_data_inventory": current_inventory,
                "dataset_artifacts": source_records,
            }
            pit_manifest = {"source_dataset": source_dataset, "source_continuous_target": source_target}
            check_true(
                "legacy_csv_only_inventory_recomputes_distinct_from_versioned_inventory",
                historical_inventory["csv_inventory_sha256"] != current_inventory["csv_inventory_sha256"],
            )
            check_true(
                "legacy_source_metadata_digest_must_match_archived_inventory_exactly",
                not source_inventory.legacy_source_inventory_semantically_equivalent(
                    test_root,
                    published={**historical_inventory, "csv_inventory_sha256": "forged"},
                    current=current_inventory,
                ),
            )
            check_true(
                "clean_legacy_pit_requires_full_csv_equivalence_proof",
                not pit_published_source_provenance_issues(
                    pit_manifest=pit_manifest, dataset_summary=current_dataset,
                    target_manifest=source_target, project_root=test_root,
                ) and "source_data_inventory mismatch" in str(pit_published_source_provenance_issues(
                    pit_manifest=pit_manifest, dataset_summary=current_dataset,
                    target_manifest=source_target,
                )),
            )
            with patch.object(
                ranking_score_store, "_load_current_validated_model_source",
                return_value=(current_dataset, source_target),
            ):
                try:
                    ranking_score_store.validate_current_pit_score_source_lineage(
                        project_root=test_root,
                        filter_id="breakout_quality_v1",
                        model_architecture="inception_time_v1",
                        experiment_profile="synthetic_profile",
                        manifest=pit_manifest,
                    )
                    official_pit_consumer_reused = True
                except ValueError:
                    official_pit_consumer_reused = False
            check_true(
                "official_rolling_pit_consumer_uses_verified_legacy_bridge",
                official_pit_consumer_reused,
            )
            check_true(
                "legacy_pit_still_rejects_changed_target_sha",
                any("source target artifacts mismatch" in issue for issue in
                    pit_published_source_provenance_issues(
                        pit_manifest=pit_manifest, dataset_summary=current_dataset,
                        target_manifest={"artifacts": {"target": {"sha256": "altered"}},
                                         "target_contract": source_target["target_contract"]},
                        project_root=test_root,
                    )),
            )
            model_dir = test_root / "historical_model"
            model_dir.mkdir()
            model_path = model_dir / "model.pt"
            model_path.write_bytes(b"unaltered historical checkpoint")
            old_fitting_source = {
                "dataset": "reduced", "dataset_policy": source_dataset["policy"],
                "source_data_inventory": historical_inventory,
                "source_dataset_artifacts": source_records,
                "source_target_artifacts": target_records_v1,
                "target_contract": source_target["target_contract"],
            }
            old_fitting_identity = {"seed": 42, "source": old_fitting_source}
            expected_fitting_identity = {
                **old_fitting_identity,
                "source": {**old_fitting_source, "source_data_inventory": current_inventory},
            }
            write_fitted_model_contract(
                model_dir=model_dir, fitting_identity=old_fitting_identity,
                selected_epoch=2, epoch_selection={}, final_refit_history=[],
                trainable_parameter_count=1, total_parameter_count=1,
            )
            old_model_manifest = {
                "source_dataset": source_dataset,
                "source_continuous_target": source_target,
                "model": build_file_manifest(model_path),
            }
            old_model_report = {
                "source_dataset": source_dataset, "target_manifest": source_target,
                "training": {"seed": 42, "selected_epoch": 2},
            }
            old_manifest_path = model_dir / "manifest.json"
            old_report_path = model_dir / "report.json"
            old_manifest_path.write_text(json.dumps(old_model_manifest), encoding="utf-8")
            old_report_path.write_text(json.dumps(old_model_report), encoding="utf-8")
            certified, certified_reason = load_reusable_fitted_model_contract(
                model_dir=model_dir, expected_identity=expected_fitting_identity,
                final_manifest_path=old_manifest_path, report_path=old_report_path,
                project_root=test_root,
            )
            without_proof, _ = load_reusable_fitted_model_contract(
                model_dir=model_dir, expected_identity=expected_fitting_identity,
                final_manifest_path=old_manifest_path, report_path=old_report_path,
            )
            check_true(
                "legacy_fitted_checkpoint_requires_original_publications_and_verified_sanitizer_equivalence",
                certified is not None and certified_reason is None and without_proof is None
                and certified.fitting_identity == old_fitting_identity,
            )
            # The archived inventory SHA and scientific fitting identity cannot
            # be rewritten; its independent content witness catches content
            # drift even when size and mtime are artificially held constant.
            import os
            csv_stamp = csv_file.stat()
            original_csv_bytes = csv_file.read_bytes()
            assert b"100" in original_csv_bytes
            csv_file.write_bytes(original_csv_bytes.replace(b"100", b"102", 1))
            os.utime(csv_file, ns=(csv_stamp.st_atime_ns, csv_stamp.st_mtime_ns))
            unchanged_metadata_inventory = source_inventory._build_source_data_inventory(
                test_root, "reduced",
            )
            try:
                source_inventory.build_source_data_inventory(test_root, "reduced")
                rejected_same_stat_content_drift = False
            except ValueError as exc:
                rejected_same_stat_content_drift = "內容已變更" in str(exc)
            csv_file.write_bytes(original_csv_bytes)
            os.utime(csv_file, ns=(csv_stamp.st_atime_ns, csv_stamp.st_mtime_ns))
            check_true(
                "source_inventory_witness_blocks_same_metadata_different_csv_content",
                unchanged_metadata_inventory == current_inventory
                and rejected_same_stat_content_drift
                and source_inventory.build_source_data_inventory(test_root, "reduced") == current_inventory,
            )
            for condition_name, affected_rows in (
                ("nonfinite_high", [*clean_rows, {**clean_rows[1], "Date": "2020-01-04", "High": float("inf")}]),
                ("duplicate_date", [*clean_rows, {**clean_rows[1], "Volume": 500}]),
            ):
                pd.DataFrame(affected_rows).to_csv(csv_file, index=False)
                changed_current = source_inventory.build_source_data_inventory(test_root, "reduced")
                changed_historical = source_inventory._build_source_data_inventory(
                    test_root, "reduced", legacy_csv_only=True,
                )
                check_true(
                    f"legacy_ohlcv_bridge_fail_closed_on_{condition_name}",
                    not source_inventory.legacy_source_inventory_semantically_equivalent(
                        test_root, published=changed_historical, current=changed_current,
                    ),
                )
            check_true(
                "legacy_ohlcv_bridge_never_accepts_other_cleaning_versions",
                not source_inventory.legacy_source_inventory_semantically_equivalent(
                    test_root, published=historical_inventory,
                    current={**current_inventory, "ohlcv_sanitization_contract_version": 3},
                ),
            )

    dataset_records = {"feature_bank": {"filename": "bank.npy", "sha256": "before"}}
    target_records = {"target_raw_r": {"filename": "target.npy", "sha256": "before"}}
    source_inventory = {"csv_inventory_sha256": "original", "ohlcv_sanitization_contract_version": 2}
    base_dataset = {
        "dataset": "full", "policy": {"window": 300},
        "source_data_inventory": source_inventory,
        "source_dataset_artifacts": dataset_records,
        "training_universe_start_date": "2011-01-01",
    }
    base_target = {"target_contract": {"target": "safety"}, "artifacts": target_records}
    summary_bundle = SimpleNamespace(summary=base_dataset, target_manifest=base_target)
    original_source = source_training_identity(summary_bundle)
    changed_source = source_training_identity(SimpleNamespace(
        summary={**base_dataset, "source_data_inventory": {**source_inventory, "csv_inventory_sha256": "after"}},
        target_manifest=base_target,
    ))
    target_changed_source = source_training_identity(SimpleNamespace(
        summary=base_dataset,
        target_manifest={**base_target, "artifacts": {"target_raw_r": {"sha256": "after"}}},
    ))
    check_true(
        "source_inventory_and_target_artifact_hashes_affect_fitting_identity",
        original_source != changed_source and original_source != target_changed_source,
    )
    evaluation_only_source = source_training_identity(SimpleNamespace(
        summary={**base_dataset, "score_eligibility_extended_to_source_tail": True},
        target_manifest={**base_target, "date_range": {"end": "2030-01-01"}},
    ))
    check_true(
        "forward_evaluation_tail_does_not_split_same_fitting_identity",
        original_source == evaluation_only_source,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "model"
        model_dir.mkdir()
        model_path = model_dir / "model.pt"
        model_path.write_bytes(b"legacy-checkpoint-verified-by-original-manifests")
        old_source = {key: value for key, value in original_source.items() if key not in (
            "source_data_inventory", "source_dataset_artifacts", "source_target_artifacts"
        )}
        old_identity = {"seed": 42, "training_settings": {"batch_size": 128}, "source": old_source}
        migrated_identity = {**old_identity, "source": original_source}
        write_fitted_model_contract(
            model_dir=model_dir, fitting_identity=old_identity,
            selected_epoch=2, epoch_selection={}, final_refit_history=[],
            trainable_parameter_count=10, total_parameter_count=12,
        )
        current_summary = {**base_dataset, "dataset_artifacts": dataset_records}
        original_manifest = {
            "source_dataset": current_summary,
            "source_continuous_target": base_target,
            "model": build_file_manifest(model_path),
        }
        original_report = {
            "source_dataset": current_summary,
            "target_manifest": base_target,
            "training": {"seed": 42, "selected_epoch": 2},
        }
        manifest_path = model_dir / "manifest.json"
        report_path = root / "report.json"
        manifest_path.write_text(json.dumps(original_manifest), encoding="utf-8")
        report_path.write_text(json.dumps(original_report), encoding="utf-8")
        compatible, reason = load_reusable_fitted_model_contract(
            model_dir=model_dir, expected_identity=migrated_identity,
            final_manifest_path=manifest_path, report_path=report_path,
        )
        check_true(
            "legacy_sidecar_can_reuse_original_checkpoint_with_dual_source_proof",
            compatible is not None and reason is None
            and compatible.fitting_identity == old_identity,
        )
        mismatched_identity = {**migrated_identity, "source": changed_source}
        stale, stale_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir, expected_identity=mismatched_identity,
            final_manifest_path=manifest_path, report_path=report_path,
        )
        check_true(
            "legacy_checkpoint_rejects_changed_inventory_even_with_unchanged_row_counts",
            stale is None and "source data/target lineage unproven" in str(stale_reason),
        )
        stale_target, target_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity={**migrated_identity, "source": target_changed_source},
            final_manifest_path=manifest_path, report_path=report_path,
        )
        check_true(
            "legacy_checkpoint_rejects_changed_target_artifact_hash",
            stale_target is None and "source data/target lineage unproven" in str(target_reason),
        )
        issues = forward_fitted_source_provenance_issues(
            model_dir=model_dir, model_manifest=original_manifest,
            report=original_report, dataset_summary=current_summary,
            target_manifest=base_target,
        )
        check_true("forward_source_lineage_matches_original_evidence", not issues)
        # Original lineage can authorize a historical FIT only together with
        # consistent fitting evidence. Neither omission nor another seed/epoch
        # may inherit the checkpoint merely because Dataset hashes agree.
        for label, training, expected_issue in (
            ("missing_seed", {"selected_epoch": 2}, "seed differs"),
            ("different_seed", {"seed": 43, "selected_epoch": 2}, "seed differs"),
            ("different_epoch", {"seed": 42, "selected_epoch": 3}, "selected_epoch differs"),
        ):
            inconsistent_report = {**original_report, "training": training}
            evidence_issues = forward_fitted_source_provenance_issues(
                model_dir=model_dir, model_manifest=original_manifest,
                report=inconsistent_report, dataset_summary=current_summary,
                target_manifest=base_target,
            )
            report_path.write_text(json.dumps(inconsistent_report), encoding="utf-8")
            rejected, rejected_reason = load_reusable_fitted_model_contract(
                model_dir=model_dir, expected_identity=migrated_identity,
                final_manifest_path=manifest_path, report_path=report_path,
            )
            check_true(
                "legacy_fitted_lineage_rejects_" + label,
                any(expected_issue in issue for issue in evidence_issues)
                and rejected is None and "source data/target lineage unproven" in str(rejected_reason),
            )
        report_path.write_text(json.dumps(original_report), encoding="utf-8")
        tampered_report = {**original_report, "source_dataset": {
            **current_summary, "source_dataset_artifacts": {"feature_bank": {"sha256": "tampered"}}
        }}
        issues = forward_fitted_source_provenance_issues(
            model_dir=model_dir, model_manifest=original_manifest,
            report=tampered_report, dataset_summary=current_summary,
            target_manifest=base_target,
        )
        check_true(
            "forward_reuse_rejects_report_source_lineage_divergence",
            any("report source_dataset_artifacts mismatch" in issue for issue in issues),
        )
        sidecar_path = model_dir / "fitted_model_manifest.json"
        sidecar_path.write_text("{broken", encoding="utf-8")
        corrupt, corrupt_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir, expected_identity=migrated_identity,
            final_manifest_path=manifest_path, report_path=report_path,
        )
        check_true(
            "corrupt_fitted_sidecar_never_falls_back_to_legacy_manifest",
            corrupt is None and "sidecar corrupt" in str(corrupt_reason),
        )

    # Direct score consumers must apply the same canonical Dataset gate as the
    # interactive Forward menu, not just verify historical score file hashes.
    from filters.breakout_quality import ranking_score_store as score_store
    from filters.breakout_quality.artifact_dependency_registry import ARTIFACT_DATASET_CORE

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset_summary_path = root / "dataset_summary.json"
        current_summary = {
            **base_dataset,
            "dataset_artifacts": dataset_records,
            "source_selection": {"requested_max_tickers": 0},
        }
        dataset_summary_path.write_text(json.dumps(current_summary), encoding="utf-8")
        publications = {"source_dataset": current_summary}
        upstream_ok = (SimpleNamespace(
            artifact_type=ARTIFACT_DATASET_CORE,
            ready=True, path=dataset_summary_path, description="READY",
        ),)
        upstream_stale = (SimpleNamespace(
            artifact_type=ARTIFACT_DATASET_CORE,
            ready=False, path=dataset_summary_path, description="inventory changed",
        ),)
        common_kwargs = {
            "project_root": root,
            "filter_id": "breakout_quality_v1",
            "model_architecture": "inception_time_v1",
            "experiment_profile": "synthetic_profile",
            "manifest": publications,
            "report": publications,
        }
        with patch(
            "filters.breakout_quality.paths.resolve_filter_output_dir",
            return_value=root,
        ), patch(
            "filters.breakout_quality.dataset_store.resolve_dataset_paths",
            return_value=SimpleNamespace(summary=dataset_summary_path),
        ), patch(
            "filters.breakout_quality.artifact_dependency_registry.collect_model_upstream_readiness",
            return_value=upstream_ok,
        ):
            score_store._validate_current_forward_score_source_lineage(**common_kwargs)
            score_store.validate_current_pit_score_source_lineage(
                **{key: value for key, value in common_kwargs.items() if key != "report"}
            )
            check_true(
                "direct_pit_score_consumer_rejects_changed_source_inventory",
                raises_expected(ValueError, lambda: score_store.validate_current_pit_score_source_lineage(
                    **{**{key: value for key, value in common_kwargs.items() if key != "report"},
                       "manifest": {"source_dataset": {
                           **current_summary,
                           "source_data_inventory": {**source_inventory, "csv_inventory_sha256": "after"},
                       }}}
                )),
            )
            tampered_manifest = {"source_dataset": {
                **current_summary,
                "source_data_inventory": {**source_inventory, "csv_inventory_sha256": "new"},
            }}
            check_true(
                "direct_forward_score_consumer_rejects_changed_source_inventory",
                raises_expected(ValueError, lambda: score_store._validate_current_forward_score_source_lineage(
                    **{**common_kwargs, "manifest": tampered_manifest}
                )),
            )
            check_true(
                "direct_forward_score_consumer_rejects_changed_report_source_artifact",
                raises_expected(ValueError, lambda: score_store._validate_current_forward_score_source_lineage(
                    **{**common_kwargs, "report": {"source_dataset": {
                        **current_summary,
                        "source_dataset_artifacts": {"feature_bank": {"sha256": "new"}},
                    }}}
                )),
            )
        with patch(
            "filters.breakout_quality.paths.resolve_filter_output_dir",
            return_value=root,
        ), patch(
            "filters.breakout_quality.dataset_store.resolve_dataset_paths",
            return_value=SimpleNamespace(summary=dataset_summary_path),
        ), patch(
            "filters.breakout_quality.artifact_dependency_registry.collect_model_upstream_readiness",
            return_value=upstream_stale,
        ):
            check_true(
                "direct_forward_score_consumer_rejects_not_ready_dataset",
                raises_expected(ValueError, lambda: score_store._validate_current_forward_score_source_lineage(
                    **common_kwargs
                )),
            )
            check_true(
                "direct_pit_score_consumer_rejects_not_ready_dataset",
                raises_expected(ValueError, lambda: score_store.validate_current_pit_score_source_lineage(
                    **{key: value for key, value in common_kwargs.items() if key != "report"}
                )),
            )

    # Regression for the Forward OOS -> Rolling fitting-identity bridge.  The
    # canonical Forward report intentionally does not own optimizer settings; those
    # live in fitted_model_manifest.json.  A valid Forward checkpoint must therefore
    # import even when report["training"] contains only scientific/result fields.
    from services.breakout_quality.point_in_time_scores import (
        _import_forward_checkpoint_to_fitting_cache,
    )
    from core.breakout_quality_registry import (
        SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        get_breakout_quality_experiment_profile,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from core.breakout_quality_runtime import CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
    # AI: Real FIT-source registration is mandatory even for synthetic checkpoints.
    fit_fixture_profile = next(
        key for key in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
        if get_breakout_quality_experiment_profile(key).training_sample_scope
           == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        and get_continuous_ranker_execution_recipe(key).context_policy.source
           == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "forward_model"
        report_dir = root / "forward_report"
        cache_dir = root / "fitting_cache"
        model_dir.mkdir()
        report_dir.mkdir()
        model_path = model_dir / "model.pt"
        model_path.write_bytes(b"forward-oos-checkpoint")
        checkpoint_record = build_file_manifest(model_path)
        model_spec = {"architecture": "synthetic_architecture"}
        experiment_settings = {
            "name": fit_fixture_profile,
            "training_sample_scope": "daily_eligible_stock_days",
        }
        execution_payload = {
            "requested_device": "cuda",
            "device_type": "cuda",
            "mixed_precision_enabled": True,
            "autocast_dtype_name": "bfloat16",
            "deterministic_algorithms": True,
            "allow_tf32": False,
        }
        plan = SimpleNamespace(as_manifest_payload=lambda: dict(execution_payload))
        source_contract = {
            "dataset_policy": "synthetic_policy",
            "dataset_storage_schema_version": 1,
            "source_data_inventory": {"rows": 1},
            "dataset_artifacts": {"dataset": {"sha256": "synthetic"}},
            "target_schema_version": 2,
            "target_contract": {"target": "synthetic"},
            "target_artifacts": {"target": {"sha256": "synthetic"}},
            "training_universe_start_date": "2011-01-01",
        }
        fold_contract = {
            "schema_version": 2,
            "fold_id": "fold_20210101_20211231",
            "filter_id": "breakout_quality_v1",
            "model_architecture": "synthetic_architecture",
            "experiment_profile": fit_fixture_profile,
            "continuous_target_id": "synthetic_target",
            "training_label_scope": "all_labels",
            "training_sample_scope": "daily_eligible_stock_days",
            "seed": 42,
            "planned_periods": {
                "validation_start": "2019-01-01",
                "validation_end": "2020-12-31",
                "score_start": "2021-01-01",
                "score_end": "2021-12-31",
            },
            "observed_periods": {
                "inner_train": {"start": "2011-01-01", "end": "2018-12-31"},
                "validation": {"start": "2019-01-01", "end": "2020-11-20"},
                "final_refit": {"start": "2011-01-01", "end": "2020-11-20"},
                "score": {"start": "2021-01-01", "end": "2021-12-31"},
            },
            "model_information_cutoff": "2020-12-31",
            "group_counts": {
                "inner_train": 100, "validation": 20, "final_refit": 120, "score": 30
            },
            "event_row_counts": {
                "inner_train": 100, "validation": 20, "final_refit": 120, "score": 30
            },
            "model_spec": model_spec,
            "experiment_settings": experiment_settings,
            "training_settings": {
                "epochs_max": 200,
                "batch_size": 128,
                "evaluation_batch_size": 4096,
                "train_prefetch_batches": 8,
                "train_prefetch_workers": 4,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 1,
                "early_stopping_min_delta": 0.0,
                "device": "cuda",
                "mixed_precision": True,
                "mixed_precision_dtype": "bfloat16",
                "deterministic_algorithms": True,
                "allow_tf32": False,
            },
            "source_contract": source_contract,
            "lookahead_contract": {
                "training_requires_label_eval_end_before_score_start": True
            },
        }
        source_manifest = {
            "filter_id": "breakout_quality_v1",
            "model_architecture": "synthetic_architecture",
            "experiment_profile": fit_fixture_profile,
            "continuous_target_id": "synthetic_target",
            "training_label_scope": "all_labels",
            "training_sample_scope": "daily_eligible_stock_days",
            "model_information_cutoff": "2020-12-31",
            "model_spec": model_spec,
            "experiment_settings": experiment_settings,
            "selected_epoch": 2,
            # Production Forward manifest.json intentionally does not own
            # torch_execution; fitted_model_manifest.json is the fitting SSOT.
            "model": checkpoint_record,
            "source_dataset": {
                "policy": source_contract["dataset_policy"],
                "dataset_storage_schema_version": source_contract["dataset_storage_schema_version"],
                "source_data_inventory": source_contract["source_data_inventory"],
                "dataset_artifacts": source_contract["dataset_artifacts"],
                "training_universe_start_date": source_contract["training_universe_start_date"],
            },
            "source_continuous_target": {
                "schema_version": source_contract["target_schema_version"],
                "target_contract": source_contract["target_contract"],
                "artifacts": source_contract["target_artifacts"],
            },
        }
        (model_dir / "manifest.json").write_text(
            json.dumps(source_manifest), encoding="utf-8"
        )
        report_path = report_dir / "continuous_ranker_report.json"
        report_path.write_text(
            json.dumps({
                "training": {"seed": 42, "selected_epoch": 2},
                "split_report": {
                    "selection_start_date": "2011-01-01",
                    "inner_validation_start_date": "2019-01-01",
                    "selection_end_date": "2020-12-31",
                    "oos_start_date": "2021-01-01",
                    "counts": {"inner_train": 100, "validation": 20, "selection": 120},
                },
                "standard_model_sop": {"evaluation_mode": "forward_oos"},
                "torch_execution": execution_payload,
                "artifacts": {"model": checkpoint_record},
            }),
            encoding="utf-8",
        )
        write_fitted_model_contract(
            model_dir=model_dir,
            fitting_identity={
                "training_settings": {
                    "max_epochs": 200,
                    "batch_size": 128,
                    "learning_rate": 0.0003,
                    "weight_decay": 0.0001,
                    "gradient_clip_norm": 1.0,
                    "use_inner_validation": True,
                    "inner_validation_months": 24,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                },
                "torch_execution": {
                    key: value for key, value in execution_payload.items()
                    if key != "requested_device"
                },
            },
            selected_epoch=2,
            epoch_selection={"best_epoch": 2},
            final_refit_history=[],
            trainable_parameter_count=1,
            total_parameter_count=1,
        )
        checkpoint = {
            "model_spec": model_spec,
            "experiment_settings": experiment_settings,
            "experiment_profile": fit_fixture_profile,
            "training_sample_scope": "daily_eligible_stock_days",
            "sequence_length": 300,
            "feature_count": 10,
            "context_count": 0,
            "selected_epoch": 2,
            "model_state_dict": {},
        }
        imported = _import_forward_checkpoint_to_fitting_cache(
            source_model_dir=model_dir,
            source_report_path=report_path,
            cache_root=cache_dir,
            fold_contract=fold_contract,
            bundle=SimpleNamespace(
                feature_bank=np.zeros((1, 300, 10), dtype=np.float32),
                group_context=np.zeros((1, 0), dtype=np.float32),
            ),
            torch_module=SimpleNamespace(load=lambda *_args, **_kwargs: checkpoint),
            execution_plan=plan,
        )
        check_true(
            "forward_oos_checkpoint_import_uses_fitted_sidecar_when_production_manifest_omits_torch_execution_and_report_omits_optimizer_fields",
            imported is not None
            and imported.get("reused_existing_cache") is False
            and Path(imported["cache_entry"]).joinpath("model.pt").read_bytes()
            == model_path.read_bytes(),
        )

        mismatched_manifest = dict(source_manifest)
        mismatched_manifest["model_spec"] = {"architecture": "wrong_architecture"}
        (model_dir / "manifest.json").write_text(
            json.dumps(mismatched_manifest), encoding="utf-8"
        )
        first_fold_mismatch_raised = False
        try:
            _import_forward_checkpoint_to_fitting_cache(
                source_model_dir=model_dir,
                source_report_path=report_path,
                cache_root=cache_dir,
                fold_contract=fold_contract,
                bundle=SimpleNamespace(
                    feature_bank=np.zeros((1, 300, 10), dtype=np.float32),
                    group_context=np.zeros((1, 0), dtype=np.float32),
                ),
                torch_module=SimpleNamespace(load=lambda *_args, **_kwargs: checkpoint),
                execution_plan=plan,
            )
        except RuntimeError as exc:
            first_fold_mismatch_raised = (
                "第一個Rolling fold應為同一fitting identity" in str(exc)
                and "manifest model_spec mismatch" in str(exc)
            )
        check_true(
            "first_rolling_fold_never_silently_refits_when_forward_candidate_claims_same_score_origin_but_identity_check_fails",
            first_fold_mismatch_raised,
        )

    from services.breakout_quality.fitted_model_artifacts import (
        fitted_model_settings_issues,
        legacy_default_fitting_settings,
        pit_fold_fitting_settings_issues,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "legacy_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"
        model_path.write_bytes(b"legacy-daily-checkpoint")
        expected_identity = {
            "filter_id": "breakout_quality_v1",
            "model_architecture": "inception_time_v1",
            "experiment_profile": "legacy_daily_profile",
            "experiment_settings": {"name": "legacy_daily_profile"},
            "model_spec": {"architecture": "inception_time_v1"},
            "training_objective": "daily_pairwise_ranking",
            "training_label_scope": "all_labels",
            "training_sample_scope": "daily_eligible_stock_days",
            "training_semantics": {"batching": "same_date"},
            "continuous_target_id": "legacy_target",
            "seed": 42,
            "training_settings": legacy_default_fitting_settings(),
            "torch_execution": {"device": "cuda", "deterministic": True},
            "split": {
                "sample_scope": "daily_eligible_stock_days",
                "selection_start_date": "2011-01-01",
                "selection_end_date": "2020-12-31",
                "inner_validation_start_date": "2019-01-01",
                "oos_start_date": "2021-01-01",
                "counts": {"inner_train": 100, "validation": 20, "selection": 120},
                "no_lookahead": True,
                "selection_label_end_before_oos": True,
                "inner_train_label_end_before_validation": True,
                "target_valid_filter_applied": True,
            },
            "source": {
                "dataset_policy": {"dataset": "full"},
                "training_universe_start_date": "2011-01-01",
                "target_contract": {"target_id": "legacy_target"},
            },
        }
        legacy_manifest = {
            key: expected_identity[key]
            for key in (
                "filter_id", "model_architecture", "experiment_profile", "experiment_settings",
                "model_spec", "training_objective", "training_label_scope",
                "training_sample_scope", "training_semantics", "continuous_target_id",
            )
        }
        legacy_manifest.update({
            "selected_epoch": 2,
            "torch_execution": {"requested_device": "cuda", **expected_identity["torch_execution"]},
            "source_continuous_target": {"target_contract": expected_identity["source"]["target_contract"]},
            "source_dataset": {
                "policy": expected_identity["source"]["dataset_policy"],
                "training_universe_start_date": expected_identity["source"]["training_universe_start_date"],
            },
            "model": build_file_manifest(model_path),
        })
        legacy_report = {
            "training": {
                "seed": 42,
                "selected_epoch": 2,
                "epoch_selection": {"best_epoch": 2},
                "final_refit_history": [{"epoch": 1}],
            },
            "torch_execution": {"requested_device": "cuda", **expected_identity["torch_execution"]},
            "split_report": {
                **{key: value for key, value in expected_identity["split"].items() if key != "counts"},
                "counts": {**expected_identity["split"]["counts"], "oos": 30},
                "oos_end_date": "2026-03-02",
            },
            "artifacts": {"model": build_file_manifest(model_path)},
        }
        manifest_path = model_dir / "manifest.json"
        report_path = root / "continuous_ranker_report.json"
        manifest_path.write_text(json.dumps(legacy_manifest), encoding="utf-8")
        report_path.write_text(json.dumps(legacy_report), encoding="utf-8")
        legacy_contract, legacy_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity=expected_identity,
            final_manifest_path=manifest_path,
            report_path=report_path,
        )
        check_true(
            "legacy_daily_forward_artifact_migrates_when_historical_fitting_defaults_match",
            legacy_contract is not None
            and legacy_reason is None
            and legacy_contract.source == "legacy_manifest_report",
        )
        changed_identity = dict(expected_identity)
        changed_settings = dict(expected_identity["training_settings"])
        changed_settings["learning_rate"] = 0.0004
        changed_identity["training_settings"] = changed_settings
        changed_contract, changed_reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity=changed_identity,
            final_manifest_path=manifest_path,
            report_path=report_path,
        )
        check_true(
            "legacy_daily_forward_artifact_does_not_mask_changed_fitting_settings",
            changed_contract is None
            and "historical defaults" in str(changed_reason),
        )
        changed_patience = dict(expected_identity["training_settings"])
        changed_patience["early_stopping_patience"] = int(changed_patience["early_stopping_patience"]) + 2
        patience_issues = fitted_model_settings_issues(
            model_dir=model_dir,
            report_path=report_path,
            expected=changed_patience,
        )
        check_true(
            "legacy_historical_fitting_defaults_do_not_follow_current_patience_change",
            legacy_default_fitting_settings()["early_stopping_patience"] == 1
            and patience_issues
            and "immutable historical defaults" in patience_issues[0],
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pit_dir = root / "point_in_time"
        fold_dir = pit_dir / "folds" / "fold_20210101_20211231"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest = {
            "training_settings": {
                "epochs_max": 200,
                "batch_size": 128,
                "learning_rate": 0.0003,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 1.0,
                "early_stopping_patience": 1,
                "early_stopping_min_delta": 0.0,
            }
        }
        (fold_dir / "manifest.json").write_text(json.dumps(fold_manifest), encoding="utf-8")
        pit_manifest = {
            "inner_validation_months": 24,
            "folds": [{"fold_id": "fold_20210101_20211231"}],
        }
        expected_settings = legacy_default_fitting_settings()
        check_true(
            "rolling_fold_current_fitting_settings_are_reusable_before_change",
            pit_fold_fitting_settings_issues(
                point_in_time_dir=pit_dir, manifest=pit_manifest, expected=expected_settings
            ) == (),
        )
        changed_settings = dict(expected_settings)
        changed_settings["early_stopping_patience"] = 3
        rolling_issues = pit_fold_fitting_settings_issues(
            point_in_time_dir=pit_dir, manifest=pit_manifest, expected=changed_settings
        )
        check_true(
            "rolling_fold_patience_change_invalidates_reuse_before_audit_or_comparison",
            any("early_stopping_patience mismatch" in issue for issue in rolling_issues),
        )

    import torch
    from filters.breakout_quality.models.factory import build_model, count_trainable_parameters
    from filters.breakout_quality.models.spec_registry import get_model_spec
    from services.breakout_quality.fitted_model_artifacts import load_model_from_fitted_checkpoint

    with tempfile.TemporaryDirectory() as temp_dir:
        model_dir = Path(temp_dir)
        spec = get_model_spec("inception_time_v1")
        feature_count = 10
        context_count = 0
        sequence_length = 300
        source_model = build_model(
            feature_count,
            context_count,
            architecture=spec.architecture,
            model_spec=spec.as_manifest_payload(),
        )

        class _SyntheticProfile:
            name = "synthetic_profile"

            @staticmethod
            def as_manifest_payload():
                return {"name": "synthetic_profile"}

        torch.save(
            {
                "model_state_dict": {
                    key: value.detach().cpu() for key, value in source_model.state_dict().items()
                },
                "feature_count": feature_count,
                "context_count": context_count,
                "sequence_length": sequence_length,
                "model_spec": spec.as_manifest_payload(),
                "experiment_profile": _SyntheticProfile.name,
                "experiment_settings": _SyntheticProfile.as_manifest_payload(),
                "selected_epoch": 2,
            },
            model_dir / "model.pt",
        )
        identity = {"synthetic": "zero_context_reload"}
        write_fitted_model_contract(
            model_dir=model_dir,
            fitting_identity=identity,
            selected_epoch=2,
            epoch_selection={"best_epoch": 2},
            final_refit_history=[],
            trainable_parameter_count=count_trainable_parameters(source_model),
            total_parameter_count=sum(int(parameter.numel()) for parameter in source_model.parameters()),
        )
        fitted_contract, _reason = load_reusable_fitted_model_contract(
            model_dir=model_dir,
            expected_identity=identity,
            final_manifest_path=model_dir / "missing_manifest.json",
            report_path=model_dir / "missing_report.json",
        )
        bundle = SimpleNamespace(
            feature_bank=np.empty((2, sequence_length, feature_count), dtype=np.float32),
            group_context=np.empty((2, context_count), dtype=np.float32),
            model_spec=spec,
            profile=_SyntheticProfile(),
        )
        loaded_model = load_model_from_fitted_checkpoint(
            torch,
            contract=fitted_contract,
            bundle=bundle,
            plan=SimpleNamespace(device=torch.device("cpu")),
        )
        check_true(
            "zero_context_daily_checkpoint_reloads_exactly_from_fitted_contract",
            fitted_contract is not None
            and count_trainable_parameters(loaded_model) == count_trainable_parameters(source_model)
            and all(
                torch.equal(source_model.state_dict()[key], loaded_model.state_dict()[key])
                for key in source_model.state_dict()
            ),
        )

    from services.research import breakout_quality_application as research_app

    # The application entrypoint validates canonical path identities.  Keep
    # the synthetic report/command evidence, but use a registered architecture
    # and profile rather than bypassing their formal path contract.
    settings = SimpleNamespace(
        filter_id="breakout_quality_v1",
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_UPPER_TAIL_PAIRWISE_PROFILE,
        seed=42,
    )
    built_contract = SimpleNamespace(report={"standard_model_sop": {}}, report_path=Path("synthetic_report.json"))
    command_args = []

    def _capture_command(command, args, **_kwargs):
        command_args.append((command, tuple(args)))
        return 0

    with tempfile.TemporaryDirectory() as temp_dir, patch.object(
        research_app, "PROJECT_ROOT", Path(temp_dir),
    ), patch.object(
        research_app,
        "_load_reusable_continuous_forward_contract",
        side_effect=[(None, "report refresh required"), (built_contract, None)],
    ), patch.object(
        research_app,
        "_collect_continuous_research_input_plan",
        return_value=SimpleNamespace(blocked=False),
    ), patch.object(
        research_app,
        "_render_continuous_research_input_plan",
        return_value=None,
    ), patch.object(
        research_app,
        "_prepare_continuous_research_inputs",
        return_value=0,
    ), patch.object(
        research_app,
        "_run_command",
        side_effect=_capture_command,
    ), patch.object(
        research_app,
        "_clear_continuous_forward_reuse_caches",
        return_value=None,
    ):
        code, contract, action = research_app._ensure_continuous_forward_model_report(
            "synthetic",
            model_id="MR-SYNTHETIC",
            settings=settings,
            prompt_for_build=False,
        )
    check_true(
        "forward_report_refresh_delegates_with_reuse_fitted_model_before_any_refit",
        code == 0
        and contract is built_contract
        and action == "BUILD/REFRESH"
        and len(command_args) == 1
        and command_args[0][0] == "train-continuous-ranker"
        and "--reuse-fitted-model" in command_args[0][1],
    )

    stale_contract = SimpleNamespace(
        seed=42,
        report={},
        report_path=Path("synthetic_report.json"),
    )
    synthetic_paths = SimpleNamespace(model_dir=Path("synthetic_model"))
    with patch.object(
        research_app,
        "load_continuous_ranker_oos_contract",
        return_value=stale_contract,
    ), patch.object(
        research_app,
        "resolve_filter_artifact_paths",
        return_value=synthetic_paths,
    ), patch.object(
        research_app,
        "fitted_model_settings_issues",
        return_value=("early_stopping_patience mismatch: artifact=1, current=3",),
    ):
        stale_loaded, stale_reason = research_app._load_reusable_continuous_forward_contract(settings)
    check_true(
        "complete_forward_report_cannot_mask_changed_early_stopping_patience",
        stale_loaded is None
        and "Fitted model training settings stale" in str(stale_reason)
        and "early_stopping_patience mismatch" in str(stale_reason),
    )

    with patch.object(
        research_app,
        "load_continuous_ranker_oos_contract",
        return_value=SimpleNamespace(
            seed=int(settings.seed),
            manifest={"source_dataset": {"dataset": "full"}},
            report={"standard_model_sop": {}},
            report_path=Path("synthetic_report.json"),
        ),
    ), patch.object(
        research_app, "resolve_filter_artifact_paths",
        return_value=SimpleNamespace(model_dir=Path("synthetic_model")),
    ), patch.object(
        research_app, "fitted_model_settings_issues", return_value=(),
    ), patch.object(
        research_app, "_forward_source_provenance_issues",
        return_value=("source_data_inventory mismatch",),
    ):
        stale_loaded, stale_reason = research_app._load_reusable_continuous_forward_contract(settings)
    check_true(
        "complete_forward_report_cannot_bypass_data_lineage_preflight",
        stale_loaded is None and "Forward fitted source lineage stale" in str(stale_reason)
        and "source_data_inventory mismatch" in str(stale_reason),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        pit_root = Path(temp_dir) / "pit"
        pit_root.mkdir()
        (pit_root / research_app.SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).write_text(
            json.dumps({
                "filter_id": settings.filter_id,
                "model_architecture": settings.model_architecture,
                "experiment_profile": settings.experiment_profile,
                "seed": 42,
                "folds": [],
            }), encoding="utf-8",
        )
        (pit_root / research_app.SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME).write_text(
            json.dumps({"status": "RESULT_AVAILABLE"}), encoding="utf-8",
        )
        with patch.object(
            research_app, "_model_robustness_dirs",
            return_value=(pit_root, pit_root),
        ), patch.object(
            research_app, "pit_fold_fitting_settings_issues",
            return_value=(),
        ), patch.object(
            rolling_source_owner, "load_selection_point_in_time_ranking_contract",
            side_effect=ValueError("source_data_inventory mismatch"),
        ), patch.object(
            rolling_source_owner, "load_selection_point_in_time_score_source_contract",
            side_effect=ValueError("source_data_inventory mismatch"),
        ):
            stale_payload, _stale_path, stale_reason = research_app._load_rolling_robustness_seed(
                settings, seed=42
            )
        check_true(
            "rolling_robustness_complete_audit_cannot_bypass_current_dataset_preflight",
            stale_payload is None and "Rolling robustness SCORE not reusable" in str(stale_reason)
            and "source_data_inventory mismatch" in str(stale_reason),
        )

    rolling_contract = SimpleNamespace(
        manifest_path=Path("synthetic_point_in_time/selection_point_in_time_manifest.json"),
        manifest={"folds": []},
        audit={},
    )
    with patch.object(
        research_app,
        "_rolling_pit_dir_for_loader",
        return_value=Path("synthetic_point_in_time"),
    ), patch.object(
        rolling_source_owner,
        "load_selection_point_in_time_ranking_contract",
        return_value=rolling_contract,
    ), patch.object(
        research_app,
        "pit_fold_fitting_settings_issues",
        return_value=("fold_20210101_20211231 early_stopping_patience mismatch: artifact=1, current=3",),
    ):
        loaded_contract, loaded_payload, rolling_reason = research_app._load_reusable_rolling_standard_report(
            settings, SimpleNamespace(point_in_time_dirname="point_in_time")
        )
    check_true(
        "complete_rolling_audit_cannot_mask_changed_early_stopping_patience",
        loaded_contract is None
        and loaded_payload is None
        and "Rolling fitted-model training settings stale" in str(rolling_reason),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "forward_seed"
        research_dir = root / "forward_report"
        model_dir.mkdir()
        research_dir.mkdir()
        model_path = model_dir / "model.pt"
        manifest_path = model_dir / "manifest.json"
        report_path = research_dir / "continuous_ranker_report.json"
        model_path.write_bytes(b"seed-model")
        manifest_path.write_text(json.dumps({
            "filter_id": settings.filter_id,
            "model_architecture": settings.model_architecture,
            "experiment_profile": settings.experiment_profile,
            "experiment_settings": get_breakout_quality_experiment_profile(
                settings.experiment_profile
            ).as_manifest_payload(),
        }), encoding="utf-8")
        report_path.write_text(json.dumps({}), encoding="utf-8")
        synthetic_contract = SimpleNamespace(
            seed=42, report_path=report_path, manifest={}, report={},
        )
        with patch.object(
            research_app, "_model_robustness_dirs", return_value=(model_dir, research_dir),
        ), patch.object(
            forward_source_owner, "load_continuous_ranker_oos_contract_at_paths",
            return_value=synthetic_contract,
        ), patch.object(
            forward_source_owner, "build_filter_artifact_paths_from_dir",
            return_value=SimpleNamespace(model_dir=model_dir),
        ), patch.object(
            forward_source_owner, "fitted_model_settings_issues",
            return_value=("early_stopping_patience mismatch: artifact=1, current=3",),
        ):
            robust_payload, _robust_path, robust_reason = research_app._load_forward_robustness_seed(
                settings, seed=42
            )
        check_true(
            "forward_robustness_seed_cannot_reuse_changed_early_stopping_patience",
            robust_payload is None
            and "Fitted model training settings stale" in str(robust_reason),
        )

        rolling_dir = root / "rolling_seed"
        rolling_dir.mkdir()
        (rolling_dir / "selection_point_in_time_manifest.json").write_text(json.dumps({
            "filter_id": settings.filter_id,
            "model_architecture": settings.model_architecture,
            "experiment_profile": settings.experiment_profile,
            "seed": 42,
            "folds": [],
        }), encoding="utf-8")
        (rolling_dir / "selection_point_in_time_audit.json").write_text(json.dumps({}), encoding="utf-8")
        with patch.object(
            research_app, "_model_robustness_dirs", return_value=(rolling_dir, research_dir)
        ), patch.object(
            rolling_source_owner, "load_selection_point_in_time_ranking_contract",
            return_value=SimpleNamespace(seed=42, manifest={"folds": []}, audit={}),
        ), patch.object(
            research_app, "pit_fold_fitting_settings_issues",
            return_value=("fold_20210101_20211231 early_stopping_patience mismatch: artifact=1, current=3",),
        ):
            rolling_robust_payload, _rolling_path, rolling_robust_reason = research_app._load_rolling_robustness_seed(
                settings, seed=42
            )
        check_true(
            "rolling_robustness_seed_cannot_reuse_changed_early_stopping_patience",
            rolling_robust_payload is None
            and "Rolling robustness fitted-model training settings stale" in str(rolling_robust_reason),
        )

    with tempfile.TemporaryDirectory() as temp_dir, patch.object(
        research_app, "PROJECT_ROOT", Path(temp_dir),
    ), patch.object(
        research_app,
        "_load_reusable_continuous_forward_contract",
        return_value=(built_contract, None),
    ), patch.object(research_app, "_run_command") as run_command:
        code, contract, action = research_app._ensure_continuous_forward_model_report(
            "synthetic",
            model_id="MR-SYNTHETIC",
            settings=settings,
            prompt_for_build=False,
        )
    check_true(
        "complete_forward_contract_reuses_without_training_or_report_rebuild",
        code == 0
        and contract is built_contract
        and action == "REUSE"
        and run_command.call_count == 0,
    )

    return _finish_case(results, summary)
