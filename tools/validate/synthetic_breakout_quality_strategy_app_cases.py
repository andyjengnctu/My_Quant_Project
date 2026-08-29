from __future__ import annotations

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    Path,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    SimpleNamespace,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    V16StrategyParams,
    _resolve_candidate_quality_ranking,
    bind_checks,
    ast,
    build_file_manifest,
    get_breakout_quality_experiment_profile,
    io,
    itertools,
    json,
    math,
    np,
    patch,
    pd,
    redirect_stdout,
    replace,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    sort_candidate_rows,
    tempfile,
)

from .source_index import read_source_ast, read_source_text
from core.display_common import FixedProgressBlock
from core.training_progress import read_trainer_pit_progress, render_training_unit_progress

def validate_breakout_quality_single_seed_single_entry_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SINGLE_SEED_SINGLE_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    project_root = Path(__file__).resolve().parents[2]
    canonical_config_path = project_root / "config" / "breakout_quality.py"
    research_config_path = project_root / "config" / "research.py"
    training_policy_path = project_root / "config" / "training_policy.py"
    canonical_source = canonical_config_path.read_text(encoding="utf-8")
    research_source = research_config_path.read_text(encoding="utf-8")
    training_policy_source = training_policy_path.read_text(encoding="utf-8")
    canonical_app_path = project_root / "services" / "research" / "breakout_quality_application.py"
    compatibility_app_path = project_root / "tools" / "filters" / "breakout_quality" / "application.py"
    canonical_app_source = canonical_app_path.read_text(encoding="utf-8")
    compatibility_app_source = compatibility_app_path.read_text(encoding="utf-8")
    strategy_app_path = project_root / "apps" / "research.py"
    strategy_config_path = project_root / "config" / "strategy_compare.py"

    check_true(
        "single_seed_contract_has_one_neutral_research_seed_ssot",
        research_source.count("RESEARCH_SINGLE_SEED =") == 1
        and "BREAKOUT_QUALITY_RANDOM_SEED = RESEARCH_SINGLE_SEED" in canonical_source
        and "OPTIMIZER_RANDOM_SEED_DEFAULT = RESEARCH_SINGLE_SEED" in training_policy_source
        and "BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED" not in canonical_source,
    )

    from config import breakout_quality as breakout_quality_config
    from config import research as research_config
    from config import training_policy as training_policy_config

    configured_seed = breakout_quality_config.resolve_breakout_quality_random_seed()
    check_true(
        "single_seed_contract_resolves_nonnegative_integer",
        isinstance(configured_seed, int) and configured_seed >= 0,
    )

    check_true(
        "single_seed_contract_model_and_optimizer_defaults_share_research_seed",
        int(breakout_quality_config.BREAKOUT_QUALITY_RANDOM_SEED)
        == int(training_policy_config.OPTIMIZER_RANDOM_SEED_DEFAULT)
        == int(research_config.RESEARCH_SINGLE_SEED),
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
    check(
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
    check_true("single_seed_contract_rejects_negative_seed", negative_seed_rejected)

    check_true(
        "model_and_strategy_apps_are_separate_entries",
        canonical_app_path.is_file()
                and compatibility_app_path.is_file()
                and strategy_app_path.is_file()
                and strategy_config_path.is_file()
                and '"strategy-compare"' not in canonical_app_source
                and "apps/research.py compare" in canonical_app_source
                and "services.research.breakout_quality_application" in compatibility_app_source,
    )

    strategy_compare_source = (
        project_root
        / "services" / "research" / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    strategy_compare_runtime_contract_source = (
        project_root
        / "services" / "research" / "strategy_compare_runtime_contract.py"
    ).read_text(encoding="utf-8")
    from config.research import get_active_model_research_provider

    provider = get_active_model_research_provider()
    research_shell_source = strategy_app_path.read_text(encoding="utf-8")
    check_true(
        "research_formal_applications_are_service_owned_and_tools_are_compatibility_only",
        provider.module == "services.research.breakout_quality_application"
        and "from services.optimizer.application import main" in research_shell_source
        and "from services.audit.runner import" in research_shell_source
        and "from services.research.strategy_compare_application import" in research_shell_source
        and "from services.research.strategy_comparison" not in research_shell_source
        and "from services.research.strategy_multi_seed_robustness" not in research_shell_source
        and "from tools.optimizer" not in research_shell_source
        and "from tools.audit" not in research_shell_source,
    )

    check_true(
        "single_seed_contract_strategy_gate_rejects_seed_mismatch",
        "resolve_strategy_compare_runtime_contract" in strategy_compare_source
        and all(
            token in strategy_compare_runtime_contract_source
            for token in (
                "int(pit_contract.seed) != int(workflow_settings.seed)",
                "Selection PIT工件seed與目前workflow不一致",
            )
        ),
    )

    summary["seed_source"] = "config.research.RESEARCH_SINGLE_SEED"
    summary["strategy_compare_entry"] = "apps/research.py compare"
    return results, summary

def validate_strategy_compare_config_driven_app_contract_case(_base_params):
    """Cross-domain Strategy Compare integration contract.

    Detailed preparation, cache/reuse, selector math, PIT, reporting and artifact
    contracts live in their dedicated validators.  This case intentionally owns
    only the integration invariants that join those domains together.
    """
    case_id = "STRATEGY_COMPARE_CONFIG_DRIVEN_APP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    project_root = Path(__file__).resolve().parents[2]

    from config import strategy_compare as strategy_config
    from config.breakout_quality import (
        get_breakout_quality_workflow_settings,
        get_continuous_ranker_execution_recipe,
    )
    from config.training_policy import (
        OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT,
        ROBUSTNESS_BENCHMARK_RESOLVED_SEEDS,
        get_robustness_benchmark_policy_snapshot,
    )
    from core.strategy_param_artifacts import (
        STRATEGY_PARAM_FAMILIES,
        resolve_strategy_param_artifact_path,
        resolve_strategy_param_benchmark_artifact_path,
    )
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_arm_runtime_dl_source_ids,
    )

    modes = strategy_config.get_strategy_rolling_test_modes()
    settings_by_mode = {
        str(mode["profile_id"]): strategy_config.get_strategy_comparison_settings(
            str(mode["profile_id"])
        )
        for mode in modes
    }
    suite_ids = {
        str(strategy_config.STRATEGY_COMPARE_PROFILES[profile_id].get("suite_id") or "")
        for profile_id in settings_by_mode
    }
    check_true(
        "current_oos_and_rolling_share_one_compare_suite",
        len(suite_ids) == 1 and "" not in suite_ids,
    )
    suite_id = next(iter(suite_ids))
    suite = strategy_config.get_strategy_compare_suite(suite_id)
    expected_arm_ids = tuple(suite["arm_ids"])
    expected_contrast_ids = tuple(suite["contrast_ids"])
    check_true(
        "current_modes_resolve_same_suite_arms_and_contrasts",
        all(
            tuple(arm.arm_id for arm in settings.enabled_arms) == expected_arm_ids
            and tuple(contrast.contrast_id for contrast in settings.enabled_contrasts) == expected_contrast_ids
            for settings in settings_by_mode.values()
        ),
    )

    from config.compatibility import strategy_compare_history as strategy_history

    check_true(
        "current_strategy_compare_catalog_is_physically_minimal_and_history_is_compatibility_only",
        set(strategy_config.STRATEGY_PARAM_SOURCES)
        == {"full_oos", "min_oos", "full_rolling", "min_rolling"}
        and set(strategy_config.STRATEGY_DL_SOURCES)
        == {"CONT13E_ROLL", "CONT13H_ROLL", "CONT13AC_ROLL", "CONT13AH_ROLL"}
        and set(strategy_config.STRATEGY_COMPARE_ARMS) == set(expected_arm_ids)
        and set(strategy_config.STRATEGY_COMPARE_CONTRASTS) == set(expected_contrast_ids)
        and {"full_roos", "min_roos", "selection_min_roos", "selection_full_roos"}
        .issubset(set(strategy_history.HISTORICAL_STRATEGY_PARAM_SOURCES))
        and {
            "CONT13E", "CONT13E_PIT", "CONT13K", "CONT13M", "CONT13M_PIT", "CONT13K_PIT",
            "CONT13K_ROLL", "CONT13M_ROLL", "CONT13P_ROLL", "CONT13Q_ROLL", "CONT13R_ROLL", "CONT13Z_ROLL",
        }.issubset(set(strategy_history.HISTORICAL_STRATEGY_DL_SOURCES))
        and {
            "C1", "C3", "C23", "C32", "C42", "C44", "C56", "C57",
            "C64", "C66", "C71", "C72", "C73", "C74", "C75",
        }.issubset(set(strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS)),
    )

    research_shell_source = (project_root / "apps" / "research.py").read_text(encoding="utf-8")
    check_true(
        "current_compare_cli_only_accepts_current_menu_profiles",
        'profile_ids = {item["profile_id"] for item in get_strategy_comparison_menu_profiles()}'
        in research_shell_source
        and 'profile_ids = {item["profile_id"] for item in get_strategy_comparison_profiles()}'
        not in research_shell_source,
    )
    check_true(
        "research_strategy_menu_exposes_oos_rolling_and_one_click_consistency_at_same_level_without_status_submenu",
        '"Extending-Window OOS"' in research_shell_source
        and '"Extending-Window Rolling"' in research_shell_source
        and '"OOS + Rolling 一鍵／Consistency"' in research_shell_source
        and '"查看目前Framework設定與工件狀態"' not in research_shell_source
        and "materialize_strategy_oos_rolling_consistency" in research_shell_source,
    )

    from services.research.strategy_compare_application import (
        build_strategy_oos_rolling_consistency,
    )

    oos_settings = settings_by_mode["extending_window_oos"]
    rolling_settings = settings_by_mode["extending_window_rolling"]
    arm_ids = [arm.arm_id for arm in oos_settings.enabled_arms]
    contrasts = {
        item.contrast_id: {
            "left": item.left,
            "right": item.right,
            "description": item.description,
        }
        for item in oos_settings.enabled_contrasts
    }
    def fake_payload(settings, *, rolling: bool):
        scenarios = {}
        geometry_arms = {}
        for idx, arm_id in enumerate(arm_ids, start=1):
            sign = -1.0 if rolling and idx == 1 else 1.0
            scenarios[arm_id] = {
                "total_return_pct": sign * idx,
                "max_drawdown_pct": 10.0 + idx,
                "return_over_max_drawdown": sign * idx / 10.0,
                "expected_value_r": sign * idx / 100.0,
                "avg_exposure_pct": 50.0 + idx,
            }
            geometry_arms[arm_id] = {
                "high_mfe_high_safety_pct": 20.0 + sign * idx,
                "high_mfe_total_pct": 40.0 + sign * idx,
            }
        return {
            "config_fingerprint": "rolling" if rolling else "oos",
            "settings": settings.as_dict(),
            "comparison_period": {"start": "2021-01-01", "end": "2026-01-01"},
            "scenarios": scenarios,
            "diagnostics": {
                "mfe_safety_geometry": {"status": "AVAILABLE", "arms": geometry_arms}
            },
            "contrasts": contrasts,
        }

    consistency = build_strategy_oos_rolling_consistency(
        fake_payload(oos_settings, rolling=False),
        fake_payload(rolling_settings, rolling=True),
    )
    check_true(
        "oos_rolling_consistency_reuses_same_suite_and_configured_contrasts",
        consistency.get("suite_id") == oos_settings.suite_id
        and len(consistency.get("contrast_rows") or [])
        == len(oos_settings.enabled_contrasts) * 7
        and {row.get("direction") for row in consistency.get("contrast_rows") or []}
        .issubset({"SAME", "DIVERGED", "FLAT", "N/A"}),
    )

    current_param_sources = {
        source.source_id: source
        for settings in settings_by_mode.values()
        for source in settings.parameter_sources.values()
        if str(source.source_id).endswith(("_oos", "_rolling"))
    }
    check_true(
        "current_parameter_sources_delegate_to_canonical_optimizer",
        bool(current_param_sources)
        and all(
            source.path_template is None
            and source.identity_manifest_path is None
            and source.builder is not None
            and str(source.builder.builder_type) == "canonical_optimizer_strategy_params"
            and not dict(source.builder.options or {})
            for source in current_param_sources.values()
        ),
    )

    physical_pairs = []
    for family in STRATEGY_PARAM_FAMILIES:
        for policy in ("base-finalist-best", "base-finalists-agree"):
            oos_path = resolve_strategy_param_artifact_path(
                Path(__file__).resolve().parents[2],
                family=family,
                evaluation_mode="oos",
                policy=policy,
            )
            rolling_path = resolve_strategy_param_artifact_path(
                Path(__file__).resolve().parents[2],
                family=family,
                evaluation_mode="rolling",
                policy=policy,
            )
            physical_pairs.append(oos_path == rolling_path)
    check_true("oos_and_rolling_share_physical_parameter_truth", all(physical_pairs))

    benchmark = get_robustness_benchmark_policy_snapshot()
    check(
        "robustness_benchmark_uses_optimizer_trial_ssot_and_fixed_seed_identity",
        (
            int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
            tuple(ROBUSTNESS_BENCHMARK_RESOLVED_SEEDS),
        ),
        (
            int(benchmark["strategy_trials_per_fold"]),
            tuple(benchmark["resolved_seeds"]),
        ),
    )
    seed = int(benchmark["resolved_seeds"][0])
    check_true(
        "benchmark_oos_and_rolling_share_per_seed_parameter_truth",
        all(
            resolve_strategy_param_benchmark_artifact_path(
                Path(__file__).resolve().parents[2],
                benchmark_id=str(benchmark["benchmark_id"]),
                seed=seed,
                family=family,
                evaluation_mode="oos",
                policy=policy,
            )
            == resolve_strategy_param_benchmark_artifact_path(
                Path(__file__).resolve().parents[2],
                benchmark_id=str(benchmark["benchmark_id"]),
                seed=seed,
                family=family,
                evaluation_mode="rolling",
                policy=policy,
            )
            for family in STRATEGY_PARAM_FAMILIES
            for policy in ("base-finalist-best", "base-finalists-agree")
        ),
    )
    current_robustness = [
        strategy_config.get_strategy_multi_seed_robustness_settings(str(mode["robustness_id"]))
        for mode in strategy_config.get_strategy_rolling_test_modes()
    ]
    check_true(
        "current_robustness_is_exact_multi_seed_form_of_compare_suite",
        all(
            tuple(item.stochastic_arm_ids) == expected_arm_ids
            and tuple(item.benchmark_strategy_arm_ids) == expected_arm_ids
            and not tuple(item.fixed_arm_ids)
            and not tuple(item.consensus_reference_arm_ids)
            and tuple(spec["contrast_id"] for spec in item.paired_contrasts) == expected_contrast_ids
            for item in current_robustness
        ),
    )
    # Completed mechanism arms remain reconstructable through the historical
    # compatibility catalog but must not continue to inflate the active matrix.
    retired_ids = {"C64", "C66", "C71", "C72", "C73", "C74", "C75", "C76"}
    check_true(
        "completed_c64_c66_c71_c75_are_historical_not_current",
        retired_ids.isdisjoint(set(expected_arm_ids))
        and retired_ids.issubset(set(strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS)),
    )

    # Current controlled comparisons keep the existing references and add two
    # MR-13AH arms: C80 copies C59's exact K/R0 contract; C81 copies C78's
    # No-K/No-R0 direct allocator.  In both cases only the DL source changes.
    expected_direct = {
        "C77": ("CONT13H_ROLL", "daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise", "inception_time_v1"),
        "C78": ("CONT13AC_ROLL", "daily_universal_predicted_upside_conditional_low_adverse_full_list_ndcg_pairwise", "inception_time_predicted_upside_context_v1"),
        "C81": ("CONT13AH_ROLL", "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise", "inception_time_v1"),
    }
    controlled_checks = []
    for current_settings in settings_by_mode.values():
        c58 = current_settings.arms["C58"]
        c59 = current_settings.arms["C59"]
        c79 = current_settings.arms["C79"]
        c80 = current_settings.arms["C80"]
        c81 = current_settings.arms["C81"]
        c59_options = dict(c59.dl_runtime_options or {})
        c79_options = dict(c79.dl_runtime_options or {})
        c80_options = dict(c80.dl_runtime_options or {})
        c78_options = dict(current_settings.arms["C78"].dl_runtime_options or {})
        c81_options = dict(c81.dl_runtime_options or {})
        ah_source = current_settings.dl_sources["CONT13AH_ROLL"]
        controlled_checks.append(
            c58.dl_enabled is False
            and c58.dl_id is None
            and c59.dl_id == "CONT13E_ROLL"
            and c79.dl_id == "CONT13AC_ROLL"
            and c80.dl_id == "CONT13AH_ROLL"
            and c59.param_source == c79.param_source == c80.param_source
            and c59.param_policy == c79.param_policy == c80.param_policy == "base-finalist-best"
            and c59.rule_policy == c79.rule_policy == c80.rule_policy == "all_off"
            and c59.dl_runtime_mode == c79.dl_runtime_mode == c80.dl_runtime_mode
            == "resource-aware-continuous-score-constrained-optimal"
            and c59_options == c79_options == c80_options
            and c80_options.get("preserve_k_r0") is True
            and c80_options.get("constrained_solver") == "exact_branch_and_bound_v1"
            and c80_options.get("selection_only") is True
            and resolve_arm_runtime_dl_source_ids(current_settings, c80) == ("CONT13AH_ROLL",)
            and c78_options == c81_options
            and c81.param_source == current_settings.arms["C78"].param_source
            and c81.param_policy == current_settings.arms["C78"].param_policy
            and c81.rule_policy == current_settings.arms["C78"].rule_policy
            and c81.dl_runtime_mode == current_settings.arms["C78"].dl_runtime_mode
            and ah_source.score_source == "selection_point_in_time"
            and ah_source.experiment_profile
            == "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"
            and ah_source.model_architecture == "inception_time_v1"
        )
        for arm_id, (dl_id, profile_name, architecture) in expected_direct.items():
            arm = current_settings.arms[arm_id]
            options = dict(arm.dl_runtime_options or {})
            source = current_settings.dl_sources[dl_id]
            controlled_checks.append(
                arm.dl_id == dl_id
                and resolve_arm_runtime_dl_source_ids(current_settings, arm) == (dl_id,)
                and arm.dl_runtime_mode == "resource-aware-continuous-score-no-k-no-r0"
                and options.get("preserve_k") is False
                and options.get("preserve_r0") is False
                and options.get("selection_order") == "model_score_desc_then_canonical_tie_v1"
                and options.get("selection_only") is True
                and source.score_source == "selection_point_in_time"
                and source.experiment_profile == profile_name
                and source.model_architecture == architecture
            )
    check_true(
        "current_controlled_contracts_include_ah_constrained_and_direct_arms",
        bool(controlled_checks) and all(controlled_checks),
    )

    check_true(
        "current_suite_schema65_adds_c80_c81_ah_conversion_matrix",
        expected_arm_ids == ("C61", "C58", "C59", "C77", "C78", "C79", "C80", "C81")
        and expected_contrast_ids == (
            "C61-C58", "C59-C58", "C77-C58",
            "C78-C58", "C78-C77",
            "C79-C58", "C79-C59", "C79-C78",
            "C80-C58", "C80-C59", "C80-C79",
            "C81-C58", "C81-C77", "C81-C78",
            "C80-C81",
        ),
    )

    check_true(
        "c79_minus_c59_is_dl_source_only_exact_kr0_control",
        all(
            settings.arms["C59"].dl_id == "CONT13E_ROLL"
            and settings.arms["C79"].dl_id == "CONT13AC_ROLL"
            and settings.arms["C59"].param_source == settings.arms["C79"].param_source
            and settings.arms["C59"].param_policy == settings.arms["C79"].param_policy
            and settings.arms["C59"].rule_policy == settings.arms["C79"].rule_policy
            and settings.arms["C59"].dl_runtime_mode == settings.arms["C79"].dl_runtime_mode
            and dict(settings.arms["C59"].dl_runtime_options or {}) == dict(settings.arms["C79"].dl_runtime_options or {})
            for settings in settings_by_mode.values()
        ),
    )

    check_true(
        "c80_minus_c59_and_c81_minus_c78_are_ah_dl_source_only_controls",
        all(
            settings.arms["C80"].dl_id == "CONT13AH_ROLL"
            and settings.arms["C59"].param_source == settings.arms["C80"].param_source
            and settings.arms["C59"].param_policy == settings.arms["C80"].param_policy
            and settings.arms["C59"].rule_policy == settings.arms["C80"].rule_policy
            and settings.arms["C59"].dl_runtime_mode == settings.arms["C80"].dl_runtime_mode
            and dict(settings.arms["C59"].dl_runtime_options or {})
            == dict(settings.arms["C80"].dl_runtime_options or {})
            and settings.arms["C81"].dl_id == "CONT13AH_ROLL"
            and settings.arms["C78"].param_source == settings.arms["C81"].param_source
            and settings.arms["C78"].param_policy == settings.arms["C81"].param_policy
            and settings.arms["C78"].rule_policy == settings.arms["C81"].rule_policy
            and settings.arms["C78"].dl_runtime_mode == settings.arms["C81"].dl_runtime_mode
            and dict(settings.arms["C78"].dl_runtime_options or {})
            == dict(settings.arms["C81"].dl_runtime_options or {})
            for settings in settings_by_mode.values()
        ),
    )


    enabled_dl_profiles = {
        str(settings.dl_sources[arm.dl_id].experiment_profile)
        for settings in settings_by_mode.values()
        for arm in settings.enabled_arms
        if arm.dl_enabled and arm.dl_id in settings.dl_sources
    }
    check_true(
        "current_dl_dependencies_are_explicitly_time_validation_authorized",
        bool(enabled_dl_profiles)
        and all(
            get_continuous_ranker_execution_recipe(profile).current_time_validation_authorized
            for profile in enabled_dl_profiles
        ),
    )

    app_source = read_source_text(Path(__file__).resolve().parents[2] / "apps" / "research.py")
    check_true(
        "research_entry_keeps_generic_strategy_compare_work_type",
        "策略組合比較" in app_source
        and "config/strategy_compare.py" in app_source
        and "MR-13" not in app_source,
    )

    settings = next(iter(settings_by_mode.values()))
    from .synthetic_breakout_quality_strategy_preparation_cases import (
        append_strategy_compare_preparation_contract_checks,
    )
    append_strategy_compare_preparation_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
        settings=settings,
        preparation_source=read_source_text(
            project_root / "services" / "research" / "strategy_compare_preparation.py"
        ),
        orchestration_source=read_source_text(
            project_root / "services" / "research" / "strategy_comparison.py"
        ),
    )

    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_completed_pair_score_reuse_contract_checks,
    )
    append_completed_pair_score_reuse_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
    )

    from core.training_scheduler import pop_next_seed_diverse_unit
    from services.research import breakout_quality_application as model_application
    from services.research import strategy_compare_training as training_runtime
    from services.research import strategy_multi_seed_robustness as robustness_runtime
    from config.research import get_active_model_research_provider

    resume_fixture_arm = settings.enabled_arms[0]
    resume_fixture_contract = {
        "resolved_seeds": [101, 202],
        "comparison_period": {"start": "2021-01-01", "end": "2022-12-31"},
        "benchmark_id": None,
    }
    with tempfile.TemporaryDirectory(prefix="robustness_partial_resume_") as temp_dir:
        resume_root = Path(temp_dir)
        resume_row = {
            "arm_id": resume_fixture_arm.arm_id,
            "seed": 101,
            "arm_order": 1,
            "seed_order": 1,
        }
        for _label, metric_key, _unit in robustness_runtime.MEAN_METRICS:
            resume_row[metric_key] = 1.0
        pd.DataFrame([resume_row]).to_csv(
            resume_root / robustness_runtime.SEED_RESULTS_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame([
            {
                "arm_id": resume_fixture_arm.arm_id,
                "seed": 101,
                "arm_order": 1,
                "seed_order": 1,
                "year": year,
                "return_pct": 1.0,
                "is_complete_year": True,
            }
            for year in (2021, 2022)
        ]).to_csv(
            resume_root / robustness_runtime.SEED_YEARLY_RESULTS_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        _existing, _existing_yearly, resume_states = (
            robustness_runtime._resolve_existing_scientific_unit_states(
                run_root=resume_root,
                contract=resume_fixture_contract,
                stochastic_arms=(resume_fixture_arm,),
                reuse_completed=True,
            )
        )
    check(
        "robustness_partial_seed_results_resume_only_missing_units",
        ("REUSE", "RUN"),
        (
            str(resume_states[(resume_fixture_arm.arm_id, 101)]["action"]),
            str(resume_states[(resume_fixture_arm.arm_id, 202)]["action"]),
        ),
    )

    application_source = read_source_text(
        project_root / "services" / "research" / "breakout_quality_application.py"
    )
    robustness_source = read_source_text(
        project_root / "services" / "research" / "strategy_multi_seed_robustness.py"
    )
    research_entry_source = read_source_text(project_root / "apps" / "research.py")
    training_contract_source = read_source_text(
        project_root / "services" / "research" / "strategy_compare_training.py"
    )
    training_process_source = read_source_text(
        project_root / "services" / "research" / "training_process.py"
    )
    execution_contract_source = read_source_text(
        project_root / "services" / "research" / "strategy_compare_execution.py"
    )
    orchestration_source = read_source_text(
        project_root / "services" / "research" / "strategy_comparison.py"
    )
    dl_artifact_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "strategy_compare_dl_artifacts.py"
    )
    pit_contract_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "strategy_compare_pit_contract.py"
    )
    pit_schedule_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "point_in_time_schedule.py"
    )
    pit_producer_source = read_source_text(
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
    )
    runtime_gate_source = read_source_text(
        project_root / "services" / "research" / "runtime_integration_gate.py"
    )
    provider = get_active_model_research_provider()
    check_true(
        "strategy_compare_model_provider_uses_one_scoped_artifact_handler",
        str(provider.strategy_artifact_handler) == "prepare_strategy_compare_artifacts"
        and not hasattr(provider, "strategy_prerequisite_handler")
        and not hasattr(provider, "strategy_upstream_handler")
        and "strategy_artifact_handler" in research_entry_source
        and "_strategy_model_artifact_scope" in research_entry_source,
    )
    check_true(
        "normal_and_robustness_share_gpu_training_scheduler_and_one_training_lifecycle_owner",
        int(strategy_config.STRATEGY_COMPARE_GPU_TRAIN_WORKERS) >= 1
        and float(strategy_config.STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS) > 0.0
        and all(
            int(strategy_config.get_strategy_multi_seed_robustness_settings(str(mode["robustness_id"])).gpu_train_workers)
            == int(strategy_config.STRATEGY_COMPARE_GPU_TRAIN_WORKERS)
            for mode in modes
        )
        and "ThreadPoolExecutor(max_workers=worker_count)" in application_source
        and "pop_next_seed_diverse_unit(jobs, futures.values())" in application_source
        and "pop_next_seed_diverse_unit(" in robustness_source
        and "run_strategy_compare_training_unit(" in application_source
        and "run_strategy_compare_training_unit(" in robustness_source
        and "build_strategy_compare_trainer_command(" not in application_source
        and "build_strategy_compare_trainer_command(" not in robustness_source
        and "run_logged_training_process(" not in application_source
        and "run_logged_training_process(" not in robustness_source
        and "audit_selection_point_in_time_scores(" not in application_source
        and "audit_selection_point_in_time_scores(" not in robustness_source
        and "def _validate_training_artifacts(" not in robustness_source
        and "def _training_command(" not in robustness_source
        and "build_strategy_compare_trainer_command(" in training_contract_source
        and "run_logged_training_process(" in training_contract_source
        and "audit_selection_point_in_time_scores(" in training_contract_source
        and "validate_strategy_compare_training_artifacts(" in training_contract_source
        and "subprocess.Popen(" not in application_source
        and "subprocess.Popen(" not in robustness_source
        and "subprocess.Popen(" in training_process_source
        and "--inner-validation-months" in training_contract_source
        and "--checkpoint-cache-root" in training_contract_source,
    )
    from filters.breakout_quality.strategy_compare_pit_contract import (
        validate_selection_pit_score_period,
    )

    auto_period_manifest = {
        "score_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "score_start_resolution": {
            "mode": "auto_earliest_legal",
            "resolved_score_start": "2021-01-01",
        },
        "evaluation_policy": {"score_end_resolution": "auto_available_end"},
        "available_history_period": {"start": "2003-09-25", "end": "2026-03-02"},
    }
    check(
        "strategy_compare_training_validator_treats_auto_bounds_as_semantic_resolution",
        ("2021-01-01", "2026-03-02"),
        validate_selection_pit_score_period(
            auto_period_manifest,
            comparison_start="auto",
            comparison_end="auto",
        ),
    )
    stale_auto_period_rejected = False
    try:
        validate_selection_pit_score_period(
            {
                **auto_period_manifest,
                "score_period": {"start": "2021-01-01", "end": "2025-12-31"},
            },
            comparison_start="2021-01-01",
            comparison_end="auto",
        )
    except ValueError:
        stale_auto_period_rejected = True
    check_true(
        "strategy_compare_training_validator_rejects_auto_end_not_equal_to_available_history_end",
        stale_auto_period_rejected,
    )

    check_true(
        "normal_and_robustness_share_one_per_arm_replay_argument_owner",
        "run_strategy_compare_active_arm(" in orchestration_source
        and "run_strategy_compare_active_arm(" in robustness_source
        and "run_strategy_compare_baseline_arm(" in orchestration_source
        and "run_strategy_compare_baseline_arm(" in robustness_source
        and "run_comparison(" not in orchestration_source
        and "run_comparison(" not in robustness_source
        and "run_standalone_baseline(" not in orchestration_source
        and "run_standalone_baseline(" not in robustness_source
        and "run_comparison(" in execution_contract_source
        and "run_standalone_baseline(" in execution_contract_source,
    )

    from services.research.strategy_compare_execution import (
        resolve_strategy_compare_ranking_options,
    )

    isolated_safety_override_checks = []
    for current_settings in settings_by_mode.values():
        for current_arm in current_settings.enabled_arms:
            options = dict(current_arm.dl_runtime_options or {})
            safety_dl_id = str(options.get("safety_dl_id") or "").strip()
            if not safety_dl_id or not current_arm.dl_id:
                continue
            primary_dl_id = str(current_arm.dl_id)
            overrides = {
                primary_dl_id: {
                    "score_path": f"isolated/{primary_dl_id}/score.csv",
                    "manifest_path": f"isolated/{primary_dl_id}/manifest.json",
                },
                safety_dl_id: {
                    "score_path": f"isolated/{safety_dl_id}/score.csv",
                    "manifest_path": f"isolated/{safety_dl_id}/manifest.json",
                },
            }
            resolved = resolve_strategy_compare_ranking_options(
                current_settings, current_arm, score_overrides=overrides
            )
            isolated_safety_override_checks.append(
                resolved.get("safety_score_path_override")
                == overrides[safety_dl_id]["score_path"]
                and resolved.get("safety_score_manifest_path_override")
                == overrides[safety_dl_id]["manifest_path"]
            )
    check_true(
        "dual_model_robustness_keeps_explicit_same_seed_safety_override_over_production_fallback",
        all(isolated_safety_override_checks)
        and "if not safety_override" in execution_contract_source,
    )
    check_true(
        "single_and_multi_seed_share_fitting_checkpoint_cache_root",
        bool(str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT).strip())
        and "multi_seed_robustness" not in str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT)
        and all(
            str(strategy_config.get_strategy_multi_seed_robustness_settings(str(mode["robustness_id"])).checkpoint_cache_root)
            == str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT)
            for mode in modes
        )
        and "STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT" in application_source,
    )

    single_seed_pending = robustness_runtime.deque([
        {"seed": 42, "dl_id": "SOURCE_B"},
        {"seed": 42, "dl_id": "SOURCE_C"},
    ])
    single_seed_selected = pop_next_seed_diverse_unit(
        single_seed_pending,
        ({"seed": 42, "dl_id": "SOURCE_A"},),
    )
    check(
        "single_seed_scheduler_uses_next_dl_source_when_second_gpu_worker_is_free",
        (42, "SOURCE_B"),
        (int(single_seed_selected["seed"]), str(single_seed_selected["dl_id"])),
    )

    rolling_settings = settings_by_mode[
        next(profile_id for profile_id in settings_by_mode if profile_id.endswith("_rolling"))
    ]
    normal_dl = next(
        rolling_settings.dl_sources[arm.dl_id]
        for arm in rolling_settings.enabled_arms
        if arm.dl_enabled and arm.dl_id in rolling_settings.dl_sources
    )
    normal_workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(normal_dl.experiment_profile)
    )
    normal_command, normal_args, _ = training_runtime.build_strategy_compare_trainer_command(
        source=normal_dl,
        workflow=normal_workflow,
        seed=int(normal_workflow.seed),
        point_in_time_dir_override=None,
        checkpoint_cache_root=(
            project_root / strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT
        ),
        resume=True,
    )
    seed_arg_index = normal_args.index("--seed") + 1
    check_true(
        "normal_parallel_training_keeps_canonical_workflow_seed_and_isolated_subprocess",
        int(normal_args[seed_arg_index]) == int(normal_workflow.seed)
        and normal_command[:3]
        == [training_runtime.sys.executable, "-m", "tools.filters.breakout_quality.build_point_in_time_scores"]
        and "--resume" in normal_args
        and "--inner-validation-months" in normal_args
        and "--checkpoint-cache-root" in normal_args
        and "--single-score-block" not in normal_args,
    )

    preparation_status_source = read_source_text(
        project_root / "services" / "research" / "strategy_compare_preparation_status.py"
    )
    comparison_source = read_source_text(
        project_root / "services" / "research" / "strategy_comparison.py"
    )
    check_true(
        "single_and_multi_seed_share_upstream_then_params_then_models_dependency_semantics",
        "resolve_planned_comparison_period_from_upstream(" in preparation_status_source
        and "execution_priority=10" in preparation_status_source
        and 'scope="upstream"' in research_entry_source
        and "scope=_strategy_model_artifact_scope(action)" in research_entry_source
        and "model phase開始前canonical upstream尚未READY" in application_source
        and "def _comparison_period_from_status(" in robustness_source
        and "status = collect_preparation_status(" in robustness_source
        and "resolve_planned_comparison_period_from_upstream(" not in robustness_source
        and "rows = [*upstream_rows, *param_rows, *benchmark_rows]" in robustness_source
        and "int(item.execution_priority)" in comparison_source,
    )

    check_true(
        "robustness_seed_generation_has_one_training_policy_owner",
        "resolve_robustness_benchmark_seeds(" in robustness_source
        and "random.Random(" not in robustness_source
        and "resolve_robustness_benchmark_seeds(" in runtime_gate_source
        and "from services.research.strategy_multi_seed_robustness import (" not in runtime_gate_source,
    )
    check_true(
        "single_and_multi_seed_share_arm_model_dependency_resolver",
        "def resolve_arm_runtime_dl_source_ids(" in dl_artifact_source
        and "def resolve_arm_artifact_dl_source_ids(" in dl_artifact_source
        and "resolve_arm_artifact_dl_source_ids(settings, arm)" in dl_artifact_source
        and "resolve_arm_runtime_dl_source_ids(settings, arm)" in robustness_source
        and "def _arm_training_dl_ids(" not in robustness_source,
    )
    check_true(
        "strategy_compare_pit_ready_contract_has_one_owner",
        "def load_validated_selection_pit_strategy_compare_contract(" in pit_contract_source
        and "def validate_selection_pit_score_period(" in pit_contract_source
        and "strategy_compare_pit_contract import" in dl_artifact_source
        and "strategy_compare_pit_contract import" in training_contract_source
        and "def _validate_selection_pit_score_period(" not in training_contract_source
        and "load_selection_point_in_time_ranking_contract(" not in training_contract_source,
    )
    check_true(
        "pit_fold_schedule_is_shared_by_producer_and_robustness_workload",
        "def build_point_in_time_fold_periods(" in pit_schedule_source
        and "build_point_in_time_fold_periods" in pit_producer_source
        and "build_point_in_time_fold_periods" in robustness_source
        and "def _pit_fold_count_for_period(" not in robustness_source,
    )
    check_true(
        "robustness_elapsed_display_reuses_common_formatter",
        "from core.display_common import FixedProgressBlock, InlineProgress, format_elapsed" in robustness_source
        and "def _format_elapsed(" not in robustness_source,
    )

    pending_trainings = robustness_runtime.deque([
        {"seed": 101, "dl_id": "SOURCE_A"},
        {"seed": 101, "dl_id": "SOURCE_B"},
        {"seed": 202, "dl_id": "SOURCE_A"},
    ])
    active_training = {object(): {"seed": 101, "dl_id": "SOURCE_A"}}
    selected_training = robustness_runtime._pop_next_training_unit(
        pending_trainings, active_training
    )
    check(
        "robustness_gpu_scheduler_prioritizes_a_different_seed_when_available",
        (202, "SOURCE_A"),
        (int(selected_training["seed"]), str(selected_training["dl_id"])),
    )

    class _TtyBuffer:
        def __init__(self):
            self.parts = []

        def isatty(self):
            return True

        def write(self, value):
            self.parts.append(str(value))
            return len(str(value))

        def flush(self):
            return None

    tty_buffer = _TtyBuffer()
    fixed_progress = FixedProgressBlock(stream=tty_buffer)
    fixed_progress.update(["seed-1 first", "seed-2 first"])
    first_render_len = len(tty_buffer.parts)
    fixed_progress.update(["seed-1 next", "seed-2 next"])
    second_render = "".join(tty_buffer.parts[first_render_len:])
    fixed_progress.clear()
    check_true(
        "strategy_compare_training_progress_redraws_fixed_worker_lines_in_place",
        "\x1b[1A" in second_render
        and "seed-1 next" in second_render
        and "seed-2 next" in second_render,
    )

    shared_unit_progress = render_training_unit_progress(
        unit_id="SOURCE_A",
        source_index=1,
        source_count=3,
        elapsed_seconds=60.1,
        pit_progress=(2, 6),
        epoch_progress=("select", 1, 200),
    )
    check_true(
        "strategy_compare_training_unit_progress_has_one_shared_canonical_format",
        "SOURCE_A 1/3" in shared_unit_progress
        and "01:00.1" in shared_unit_progress
        and "PIT 2/6" in shared_unit_progress
        and "remain 4" in shared_unit_progress
        and "active fold 3/6" in shared_unit_progress
        and "epoch select 1/200" in shared_unit_progress,
    )

    with tempfile.TemporaryDirectory(prefix="strategy_progress_log_") as temp_dir:
        progress_log = Path(temp_dir) / "train.log"
        progress_log.write_text(
            "__BQ_PIT_PROGRESS__ completed=2 total=6 active=3\n"
            "__BQ_EPOCH_PROGRESS__ phase=select epoch=1/200\n",
            encoding="utf-8",
        )
        check(
            "strategy_compare_single_and_multi_seed_share_pit_fold_progress_parser",
            (2, 6),
            read_trainer_pit_progress(progress_log),
        )
        # The machine marker must remain sufficient even when the verbose PIT plan
        # is absent from the bounded log window.
        progress_log.write_text(
            "__BQ_PIT_PROGRESS__ completed=6 total=6 active=0\n",
            encoding="utf-8",
        )
        check(
            "strategy_compare_pit_fold_progress_parser_marks_completed_plan",
            (6, 6),
            read_trainer_pit_progress(progress_log),
        )

    with tempfile.TemporaryDirectory(prefix="robustness_atomic_retry_") as temp_dir:
        target = Path(temp_dir) / "manifest.json"
        target.write_text("old\n", encoding="utf-8")
        import core.file_integrity as file_integrity

        original_replace = file_integrity.os.replace
        replace_attempts = {"count": 0}

        def transient_replace(src, dst):
            replace_attempts["count"] += 1
            if replace_attempts["count"] <= 2:
                raise PermissionError("simulated transient manifest lock")
            return original_replace(src, dst)

        with patch.object(file_integrity.os, "replace", side_effect=transient_replace), patch.object(
            file_integrity.time, "sleep", return_value=None
        ):
            file_integrity.atomic_write_text(target, "new\n")

        check(
            "robustness_progress_manifest_retries_transient_permission_error",
            (3, "new", 0),
            (
                replace_attempts["count"],
                target.read_text(encoding="utf-8").strip(),
                len(list(Path(temp_dir).glob(".manifest.json.*.tmp"))),
            ),
        )

    summary.update(
        {
            "suite_id": suite_id,
            "mode_count": len(settings_by_mode),
            "arm_count": len(expected_arm_ids),
            "contrast_count": len(expected_contrast_ids),
            "current_dl_profiles": sorted(enabled_dl_profiles),
        }
    )
    return results, summary

def validate_mr13z_c75_conversion_contract_case(_base_params):
    """Protect the first PIT-safe direct Joint-Min strategy conversion after MR-13Z GO."""

    case_id = "MR13Z_C75_CONVERSION_CONTRACT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_continuous_ranker_research_spec,
    )
    from config.strategy_compare import (
        STRATEGY_COMPARE_SCHEMA_VERSION,
        STRATEGY_DL_SOURCES,
        get_strategy_compare_suite,
    )
    from config.compatibility.strategy_compare_history import (
        HISTORICAL_STRATEGY_COMPARE_ARMS,
        HISTORICAL_STRATEGY_COMPARE_CONTRASTS,
        HISTORICAL_STRATEGY_DL_SOURCES,
    )
    from filters.breakout_quality.ranking_score_store import PIT_OPTIONAL_SCORE_COLUMNS
    from filters.breakout_quality.strategy_score_projection import runtime_score_projection_requirements

    profile_name = DAILY_UNIVERSAL_SAFETY_RAW_MFE_JOINT_MIN_PATCH_TRANSFORMER_ATTN_POOL_MLP_HEAD_FULL_LIST_NDCG_PAIRWISE_PROFILE
    spec = get_continuous_ranker_research_spec(profile_name)
    check(
        "mr13z_conversion_authorization_is_research_only",
        ("MR-13Z", True, True),
        (spec.model_research_id, spec.selection_pit_authorized, spec.current_time_validation_authorized),
    )

    source = dict(HISTORICAL_STRATEGY_DL_SOURCES.get("CONT13Z_ROLL") or {})
    check(
        "cont13z_roll_is_mr13z_selection_pit_source",
        (profile_name, "patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1", "selection_point_in_time"),
        (source.get("experiment_profile"), source.get("model_architecture"), source.get("score_source")),
    )
    builder = dict(source.get("forward_scores_builder") or {})
    check_true(
        "strategy_compare_cannot_train_missing_mr13z_folds",
        builder.get("enabled") is True
        and builder.get("builder_type") == "selection_pit_from_existing_folds",
    )

    check_true(
        "mr13z_pit_contract_can_preserve_raw_mfe_primary_and_joint_min_auxiliary",
        "raw_safety_score" in PIT_OPTIONAL_SCORE_COLUMNS
        and "raw_mfe_score" in PIT_OPTIONAL_SCORE_COLUMNS
        and "joint_min_score" in PIT_OPTIONAL_SCORE_COLUMNS,
    )

    c75 = dict(HISTORICAL_STRATEGY_COMPARE_ARMS.get("C75") or {})
    options = dict(c75.get("dl_runtime_options") or {})
    check(
        "c75_is_direct_joint_min_no_k_no_r0_selection_only",
        (
            "CONT13Z_ROLL",
            "resource-aware-continuous-score-no-k-no-r0",
            False, False,
            "model_score_desc_then_canonical_tie_v1",
            "joint_min_score",
            True,
        ),
        (
            c75.get("dl_id"), c75.get("dl_runtime_mode"),
            options.get("preserve_k"), options.get("preserve_r0"),
            options.get("selection_order"), options.get("primary_score_column"),
            options.get("selection_only"),
        ),
    )
    check_true(
        "c75_has_no_threshold_product_or_calibration_treatment",
        "safety_gate" not in options
        and "safety_percentile_cutoff" not in options
        and "joint_score_transform" not in options
        and "calibration" not in options
        and "score_weight" not in options,
    )

    suite = get_strategy_compare_suite("extending_current")
    check(
        "current_suite_schema65_retires_c75_but_keeps_history",
        (65, 0, 8),
        (int(STRATEGY_COMPARE_SCHEMA_VERSION), tuple(suite.get("arm_ids") or ()).count("C75"), len(tuple(suite.get("arm_ids") or ()))),
    )
    check_true(
        "c75_primary_and_secondary_contrasts_are_historical",
        dict(HISTORICAL_STRATEGY_COMPARE_CONTRASTS.get("C75-C74") or {}).get("left") == "C75"
        and dict(HISTORICAL_STRATEGY_COMPARE_CONTRASTS.get("C75-C74") or {}).get("right") == "C74"
        and dict(HISTORICAL_STRATEGY_COMPARE_CONTRASTS.get("C75-C71") or {}).get("left") == "C75"
        and dict(HISTORICAL_STRATEGY_COMPARE_CONTRASTS.get("C75-C71") or {}).get("right") == "C71",
    )

    requirements = runtime_score_projection_requirements(
        settings_payload={"dl_sources": {**STRATEGY_DL_SOURCES, **HISTORICAL_STRATEGY_DL_SOURCES}},
        on_arm_payload=c75,
    )
    check(
        "c75_cache_projection_hashes_only_consumed_joint_min_score",
        {"CONT13Z_ROLL": ("joint_min_score",)},
        requirements,
    )
    check_true(
        "c75_is_not_robustness_or_production_promotion",
        c75.get("robustness_role") == "off"
        and source.get("score_source") == "selection_point_in_time",
    )

    summary["training_performed"] = False
    return results, summary


def validate_breakout_quality_daily_pit_strategy_runtime_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_DAILY_PIT_STRATEGY_RUNTIME"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config import strategy_compare as strategy_config
    from core.extended_signals import attach_breakout_quality_rank
    from filters.breakout_quality.ranker_sample_contract import (
        build_score_eligibility_contract,
    )
    from filters.breakout_quality.ranking_score_store import (
        _validate_score_eligibility_contract,
        load_continuous_ranker_oos_contract,
        load_continuous_ranker_oos_score_table,
        lookup_continuous_ranker_oos_candidate_score,
        resolve_continuous_ranker_oos_score_path,
    )
    from filters.breakout_quality.runtime import (
        breakout_quality_ranking_source_context,
        resolve_breakout_quality_candidate_rank,
    )
    from tools.filters.breakout_quality.build_point_in_time_scores import (
        _fold_group_ids,
    )

    daily_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    event_profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    daily_contract = build_score_eligibility_contract(daily_profile)
    event_contract = build_score_eligibility_contract(event_profile)
    current_daily_accepted = True
    try:
        _validate_score_eligibility_contract(
            {"score_eligibility_contract": daily_contract}, profile=daily_profile
        )
    except ValueError:
        current_daily_accepted = False
    legacy_daily_rejected = False
    try:
        _validate_score_eligibility_contract({}, profile=daily_profile)
    except ValueError as exc:
        legacy_daily_rejected = "重新建立PIT Scores" in str(exc)
    check(
        "daily_pit_score_presence_depends_only_on_past_feature_history_contract",
        (
                    "feature_history_only", False, True,
                    "canonical_breakout_event_membership", False,
                    True, True,
                ),
        (
                    daily_contract["eligibility_basis"],
                    daily_contract["future_target_required_for_score"],
                    daily_contract["target_valid_required_for_training"],
                    event_contract["eligibility_basis"],
                    event_contract["future_target_required_for_score"],
                    current_daily_accepted,
                    legacy_daily_rejected,
                ),
    )

    group_table = pd.DataFrame([
        {"ticker": "A", "date": "2010-01-01", "group_index": 0, "label": -1, "label_eval_end_date": "2010-02-01"},
        {"ticker": "B", "date": "2012-01-02", "group_index": 1, "label": -1, "label_eval_end_date": "2012-02-15"},
        {"ticker": "C", "date": "2013-12-01", "group_index": 2, "label": -1, "label_eval_end_date": "2013-12-20"},
        {"ticker": "D", "date": "2014-01-02", "group_index": 3, "label": -1, "label_eval_end_date": pd.NaT},
        {"ticker": "E", "date": "2014-06-01", "group_index": 4, "label": -1, "label_eval_end_date": pd.NaT},
    ])
    bundle = SimpleNamespace(
        group_table=group_table,
        profile=SimpleNamespace(
            training_label_scope=TRAINING_LABEL_SCOPE_ALL,
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        ),
        target_valid=np.array([True, True, False, False, False], dtype=bool),
        event_group_index=np.arange(len(group_table), dtype=np.int64),
    )
    ids = _fold_group_ids(
        bundle,
        {
            "fold_id": "fold_20140101_20141231",
            "score_start": pd.Timestamp("2014-01-01"),
            "score_end": pd.Timestamp("2014-12-31"),
        },
        validation_months=24,
    )
    check(
        "daily_pit_target_invalid_rows_are_excluded_from_training_but_still_scored",
        ([0], [1], [0, 1], [3, 4]),
        (
                    ids["train_ids"].tolist(), ids["validation_ids"].tolist(),
                    ids["final_ids"].tolist(), ids["score_ids"].tolist(),
                ),
    )

    lookup_dates = []

    def _fake_lookup(**kwargs):
        date = pd.Timestamp(kwargs["signal_date"]).strftime("%Y-%m-%d")
        lookup_dates.append(date)
        return {
            "score": 0.8,
            "available": True,
            "score_date": date,
            "score_source": "selection_point_in_time",
        }

    with patch(
        "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
        side_effect=_fake_lookup,
    ):
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_v1",
            experiment_profile=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        ):
            resolve_breakout_quality_candidate_rank(
                ticker="2330", signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"), high_len=275,
            )
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_v1",
            experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        ):
            resolve_breakout_quality_candidate_rank(
                ticker="2330", signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"), high_len=275,
            )
    check(
        "mr12b_keeps_event_score_date_while_mr13a_refreshes_latest_completed_information_date",
        ["2020-01-02", "2020-01-09"],
        lookup_dates,
    )

    dual_head_lookup_columns = []

    def _fake_dual_head_lookup(**kwargs):
        dual_head_lookup_columns.append(kwargs.get("score_column"))
        date = pd.Timestamp(kwargs["signal_date"]).strftime("%Y-%m-%d")
        return {
            "score": 0.82 if kwargs.get("score_column") is None else 0.74,
            "available": True,
            "score_date": date,
            "score_source": "selection_point_in_time",
        }

    with patch(
        "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
        side_effect=_fake_dual_head_lookup,
    ):
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_conditional_mfe_safety_v1",
            experiment_profile="daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise",
            ranking_options={
                "safety_dl_id": "CONT13P_ROLL",
                "safety_filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                "safety_score_source": "selection_point_in_time",
                "safety_model_architecture": "inception_time_conditional_mfe_safety_v1",
                "safety_experiment_profile": "daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise",
                "safety_score_column": "conditional_safety_score",
            },
        ):
            dual_head_payload = resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date=pd.Timestamp("2021-01-05"),
                information_date=pd.Timestamp("2021-01-04"),
                high_len=275,
            )
    check(
        "mr13p_runtime_reads_primary_and_conditional_safety_from_explicit_columns_of_one_source",
        ([None, "conditional_safety_score"], 0.82, 0.74, "CONT13P_ROLL"),
        (
            dual_head_lookup_columns,
            float(dual_head_payload["score"]),
            float(dual_head_payload["safety_score"]),
            dual_head_payload["safety_dl_id"],
        ),
    )

    daily_params = replace(
        _base_params,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    daily_state = {}
    attach_breakout_quality_rank(daily_state, {
        "score": 0.3,
        "available": True,
        "score_date": "2020-01-02",
        "score_source": "selection_point_in_time",
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": "inception_time_v1",
        "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    })
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            return_value={
                "score": 0.9, "available": True, "score_date": "2020-01-09",
                "score_source": "selection_point_in_time",
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                "model_architecture": "inception_time_v1",
                "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
            },
        ) as lookup:
            refreshed = _resolve_candidate_quality_ranking(
                params=daily_params,
                ticker="2330",
                signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"),
                signal_state=daily_state,
            )
    check(
        "daily_continuation_recomputes_rank_each_day_instead_of_inheriting_event_score",
        (1, 0.9, "2020-01-09"),
        (lookup.call_count, refreshed["score"], refreshed["score_date"]),
    )

    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.paths import (
        resolve_filter_artifact_paths,
        resolve_filter_model_output_dir,
    )

    daily_forward_contract_accepts_canonical_gzip = False
    daily_forward_table_uses_all_oos_rows_without_split_column = False
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        filter_id = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
        architecture = "inception_time_v1"
        profile_name = DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
        artifact_paths = resolve_filter_artifact_paths(
            root, filter_id, architecture, profile_name
        )
        output_dir = resolve_filter_model_output_dir(
            root, filter_id, architecture, profile_name
        )
        artifact_paths.model_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths.model_path.write_bytes(b"synthetic-daily-model")
        score_path = resolve_continuous_ranker_oos_score_path(
            root, filter_id, architecture, profile_name
        )
        pd.DataFrame([
            {
                "ticker": "0050", "date": "2021-01-04", "group_index": 1,
                "target_raw_r": 0.10, "target_daily_percentile": 0.5,
                "model_score": 0.55,
            },
            {
                "ticker": "2330", "date": "2021-01-04", "group_index": 2,
                "target_raw_r": 0.20, "target_daily_percentile": 1.0,
                "model_score": 0.75,
            },
            {
                "ticker": "2317", "date": "2021-01-05", "group_index": 3,
                "target_raw_r": np.nan, "target_daily_percentile": np.nan,
                "model_score": 0.65,
            },
        ]).to_csv(
            score_path, index=False, encoding="utf-8-sig", compression="gzip"
        )
        pairwise_contract = {
            "pair_scope": "same_date_non_tied_target_pairs",
            "pair_weighting": "equal_pair_weight",
            "model_margin": "pass_logit_minus_reject_logit",
            "batching": "whole_date_pack_no_date_split",
            "runtime_score": "softmax_pass_probability",
        }
        report = {
            "status": "RESULT_AVAILABLE_PENDING_REVIEW",
            "filter_id": filter_id,
            "model_architecture": architecture,
            "experiment_profile": profile_name,
            "experiment_settings": daily_profile.as_manifest_payload(),
            "training": {
                "objective": daily_profile.training_objective,
                "loss": daily_profile.loss_name,
                "sample_scope": daily_profile.training_sample_scope,
                "batching": pairwise_contract["batching"],
                "pairwise_contract": pairwise_contract,
                "seed": 42,
            },
            "score_eligibility_contract": daily_contract,
            "forward_score_coverage": {
                "inference_eligible_groups": 3,
                "target_evaluable_groups": 2,
                "future_target_required_for_score": False,
            },
            "artifacts": {
                "model": build_file_manifest(artifact_paths.model_path),
                "oos_scores_gzip": build_file_manifest(score_path),
            },
        }
        (output_dir / "continuous_ranker_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        manifest = {
            "filter_id": filter_id,
            "model_architecture": architecture,
            "experiment_profile": profile_name,
            "experiment_settings": daily_profile.as_manifest_payload(),
            "training_objective": daily_profile.training_objective,
            "training_sample_scope": daily_profile.training_sample_scope,
            "continuous_target_id": daily_profile.continuous_target_id,
            "model": build_file_manifest(artifact_paths.model_path),
            "score_eligibility_contract": daily_contract,
            "forward_score_coverage": {
                "inference_eligible_groups": 3,
                "target_evaluable_groups": 2,
                "future_target_required_for_score": False,
            },
            "research_outputs": {
                "oos_scores_gzip": build_file_manifest(score_path),
            },
            "outer_oos_policy": {
                "oos_start_date": "2021-01-01",
                "effective_oos_end_date": "2021-12-31",
            },
            "model_information_cutoff": "2020-12-31",
        }
        artifact_paths.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        load_continuous_ranker_oos_contract.cache_clear()
        load_continuous_ranker_oos_score_table.cache_clear()
        forward_contract = load_continuous_ranker_oos_contract(
            str(root), filter_id, architecture, profile_name
        )
        forward_table = load_continuous_ranker_oos_score_table(
            str(root), filter_id, architecture, profile_name
        )
        daily_forward_contract_accepts_canonical_gzip = (
            forward_contract.score_path.name == "daily_ranker_oos_scores.csv.gz"
            and forward_contract.continuous_target_id
            == daily_profile.continuous_target_id
        )
        daily_forward_table_uses_all_oos_rows_without_split_column = (
            len(forward_table) == 3
            and "split" not in forward_table.columns
            and ("2330", "2021-01-04") in forward_table.index
            and ("2317", "2021-01-05") in forward_table.index
            and pd.isna(forward_table.loc[("2317", "2021-01-05"), "target_raw_r"])
        )
        daily_lookup_uses_precomputed_score_period = False
        try:
            with patch.object(
                pd.MultiIndex,
                "get_level_values",
                side_effect=AssertionError("lookup must not rescan the full score index"),
            ):
                lookup_payload = lookup_continuous_ranker_oos_candidate_score(
                    project_root=str(root),
                    ticker="2330",
                    signal_date="2021-01-04",
                    filter_id=filter_id,
                    model_architecture=architecture,
                    experiment_profile=profile_name,
                    score_path_override=str(score_path),
                )
            daily_lookup_uses_precomputed_score_period = (
                lookup_payload["available"]
                and float(lookup_payload["score"]) == 0.75
                and forward_table.attrs.get("available_from") == "2021-01-04"
                and forward_table.attrs.get("available_through") == "2021-01-05"
            )
        except AssertionError:
            daily_lookup_uses_precomputed_score_period = False
    check(
        "mr13a_forward_oos_contract_uses_daily_gzip_artifact_not_event_csv_schema",
        (True, True, True),
        (
                    daily_forward_contract_accepts_canonical_gzip,
                    daily_forward_table_uses_all_oos_rows_without_split_column,
                    daily_lookup_uses_precomputed_score_period,
                ),
    )

    settings = strategy_config.get_strategy_comparison_settings()
    c24, c25, c27, c28 = (
        settings.arms["C24"], settings.arms["C25"],
        settings.arms["C27"], settings.arms["C28"],
    )
    selection_source = settings.dl_sources["CONT13A_PIT"]
    check(
        "stage3_selection_daily_pit_arm_definitions_remain_frozen_after_gate",
        (
                    False, False, "CONT13A_PIT", "CONT13A_PIT",
                    c24.param_source, c24.rule_policy, c24.dl_runtime_mode,
                    c25.param_source, c25.rule_policy, c25.dl_runtime_mode,
                    "selection_point_in_time", DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
                ),
        (
                    c27.enabled, c28.enabled, c27.dl_id, c28.dl_id,
                    c27.param_source, c27.rule_policy, c27.dl_runtime_mode,
                    c28.param_source, c28.rule_policy, c28.dl_runtime_mode,
                    selection_source.score_source, selection_source.experiment_profile,
                ),
    )

    forward_settings = strategy_config.get_strategy_comparison_settings("forward_oos")
    c1, c3, c20, c29, c30, c31 = (
        forward_settings.arms["C1"], forward_settings.arms["C3"], forward_settings.arms["C20"],
        forward_settings.arms["C29"], forward_settings.arms["C30"], forward_settings.arms["C31"],
    )
    forward_source = forward_settings.dl_sources["CONT13A"]
    active_contrasts = tuple(
        item.contrast_id for item in forward_settings.enabled_contrasts
    )
    configured_forward_arms = tuple(
        strategy_config.STRATEGY_COMPARE_PROFILES["forward_oos"]["arm_ids"]
    )
    configured_forward_contrasts = tuple(
        strategy_config.STRATEGY_COMPARE_PROFILES["forward_oos"]["contrast_ids"]
    )
    check_true(
        "forward_oos_profile_honors_config_and_freezes_min_mr12b_mr13a_runtime_identity",
        {arm.arm_id for arm in forward_settings.enabled_arms} == set(configured_forward_arms)
                and set(active_contrasts) == set(configured_forward_contrasts)
                and (c20.param_source, c20.rule_policy, c20.dl_id, c20.dl_runtime_mode)
                == (
                    "min_roos", "all_off", "CONT12B",
                    "resource-aware-continuous-max-dl-feasible-ascent",
                )
                and (c29.param_source, c29.rule_policy, c29.dl_id, c29.dl_runtime_mode)
                == (
                    "min_roos", "all_off", "CONT13A",
                    "resource-aware-continuous-max-dl-feasible-ascent",
                )
                and forward_source.score_source == "continuous_ranker_oos"
                and forward_source.experiment_profile == DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
                and c30.param_source == "full_roos"
                and c31.param_source == "full_roos",
    )
    check_true(
        "forward_oos_common_period_remains_auto_resolved_from_enabled_dl_sources",
        c1.param_source == "full_roos" and not c1.dl_enabled
                and c3.param_source == "min_roos" and not c3.dl_enabled
                and forward_settings.start_date is None and forward_settings.end_date is None,
    )

    project_root = Path(__file__).resolve().parents[2]
    strategy_diagnostics_source = (
        project_root / "filters" / "breakout_quality" / "strategy_compare_diagnostics.py"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        project_root / "services" / "breakout_quality" /
        "continuous_ranker_pipeline.py"
    ).read_text(encoding="utf-8")
    check_true(
        "training_and_strategy_diagnostics_share_domain_layer_profile_sample_provider",
        "load_profile_continuous_ranker_data" in strategy_diagnostics_source
                and "load_profile_continuous_ranker_data" in pipeline_source,
    )

    summary["profile"] = DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    summary["stage"] = "selection_strategy_translation"
    summary["future_target_used_for_score_presence"] = False
    return results, summary


def validate_breakout_quality_runtime_integration_gate_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_INTEGRATION_GATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        get_breakout_quality_workflow_settings,
    )
    from config.strategy_compare import get_strategy_runtime_integration_settings
    from services.research.strategy_compare_application import (
        dispatch_runtime_integration_action,
        runtime_integration_execution_enabled,
    )
    from core.strategy_comparison import (
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    )
    from services.research.runtime_integration_gate import (
        _exact_certificate_status,
        _overall_decision,
        resolve_runtime_candidate_contract,
    )

    cfg = get_strategy_runtime_integration_settings()
    workflow = get_breakout_quality_workflow_settings()
    workflow_profile = get_breakout_quality_experiment_profile(workflow.experiment_profile)
    contract = resolve_runtime_candidate_contract()
    selection = dict(contract["selection"])
    forward = dict(contract["forward"])
    runtime = dict(contract["runtime"])
    selection_source = dict(selection["dl_source"])
    forward_source = dict(forward["dl_source"])

    check_true(
        "runtime_integration_candidate_is_same_mr13e_exact_semantics_across_stages",
        selection_source["experiment_profile"]
                    == forward_source["experiment_profile"]
                    == workflow.experiment_profile
                    and workflow_profile.training_sample_scope
                    == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                    and selection_source["score_source"] == "selection_point_in_time"
                    and forward_source["score_source"] == "continuous_ranker_oos"
                    and runtime["param_source"] == forward["param_source"]
                    and selection["rule_policy"] == forward["rule_policy"] == "all_off"
                    and selection["dl_runtime_mode"]
                    == forward["dl_runtime_mode"]
                    == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
                    and dict(selection["dl_runtime_options"]).get("preserve_k_r0") is True
                    and dict(forward["dl_runtime_options"]).get("preserve_k_r0") is True
                    and dict(selection["dl_runtime_options"]).get("constrained_solver")
                    == dict(forward["dl_runtime_options"]).get("constrained_solver")
                    == "exact_branch_and_bound_v1"
                    and dict(selection["dl_runtime_options"]).get("selection_only") is True
                    and dict(forward["dl_runtime_options"]).get("selection_only") is False,
    )
    check_true(
        "runtime_integration_legacy_gate_is_not_executable_from_current_research_while_anchor_identity_is_preserved",
        not runtime_integration_execution_enabled()
                    and runtime["experiment_profile"] == workflow.experiment_profile
                    and cfg.comparison_anchor_experiment_profile != workflow.experiment_profile
                    and selection["candidate_arm_id"] == cfg.selection_candidate_arm_id
                    and forward["candidate_arm_id"] == cfg.forward_candidate_arm_id,
    )
    from filters.breakout_quality.runtime import get_breakout_quality_ranking_source_context

    runtime_context = get_breakout_quality_ranking_source_context()
    check_true(
        "promoted_workflow_runtime_uses_mr13e_exact_canonical_context",
        workflow.runtime_strategy_enabled
                    and runtime_context.experiment_profile == workflow.experiment_profile
                    and runtime_context.model_architecture == workflow.model_architecture
                    and runtime_context.score_source == "canonical_runtime"
                    and runtime_context.ranking_policy == workflow.runtime_ranking_policy
                    and dict(runtime_context.ranking_options or {}) == dict(workflow.runtime_ranking_options),
    )
    from services.breakout_quality.export_scores import parse_args as parse_score_export_args

    parsed_runtime_export = parse_score_export_args(
        [
            "--experiment-profile",
            workflow.experiment_profile,
            "--scope",
            "workflow_runtime",
        ]
    )
    check(
        "promoted_mr13e_profile_is_accepted_by_workflow_runtime_score_export_cli",
        (workflow.experiment_profile, "workflow_runtime"),
        (parsed_runtime_export.experiment_profile, parsed_runtime_export.scope),
    )

    from services.breakout_quality.export_scores import run_export as run_score_export

    with patch(
        "services.breakout_quality.export_scores._run_daily_continuous_workflow_export",
        return_value=0,
    ) as daily_export:
        dispatch_rc = run_score_export(
            project_root=Path(__file__).resolve().parents[2],
            argv=[
                "--experiment-profile",
                workflow.experiment_profile,
                "--scope",
                "workflow_runtime",
            ],
        )
    check_true(
        "promoted_mr13e_workflow_export_dispatches_to_daily_continuous_path",
        dispatch_rc == 0 and daily_export.call_count == 1,
    )

    export_source = (
        Path(__file__).resolve().parents[2]
        / "services" / "breakout_quality" / "export_scores.py"
    ).read_text(encoding="utf-8")
    check_true(
        "workflow_runtime_scores_are_separate_from_forward_oos_canonical_scores",
        '"runtime"' in export_source
                    and '"runtime_manifest.json"' in export_source
                    and "if args.scope == RUNTIME_SCOPE_WORKFLOW" in export_source
                    and "canonical Forward-OOS research score/manifest are not modified" in export_source,
    )
    check_true(
        "workflow_runtime_mr13e_export_uses_daily_continuous_ranker_contract",
        "load_daily_universal_ranker_data" in export_source
                    and "load_continuous_ranker_oos_contract" in export_source
                    and "ranker_api.predict_scores" in export_source
                    and '"model_score"' in export_source,
    )

    from filters.breakout_quality import workflow_runtime_score_store as runtime_store

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        models_root = temp_root / "models"
        source_model = temp_root / "canonical" / "model.pt"
        source_model.parent.mkdir(parents=True, exist_ok=True)
        source_model.write_bytes(b"synthetic-mr13e-model")
        runtime_dir = (
            models_root
            / "runtime"
            / "breakout_quality"
            / BREAKOUT_QUALITY_DEFAULT_FILTER_ID
            / workflow.model_architecture
            / workflow.experiment_profile
        )
        runtime_dir.mkdir(parents=True, exist_ok=True)
        score_path = runtime_dir / "scores.csv"
        unavailable_path = runtime_dir / "unavailable_scores.csv"
        manifest_path = runtime_dir / "runtime_manifest.json"
        pd.DataFrame(
            [
                {"ticker": "2330", "date": "2026-08-14", "model_score": 0.75},
            ]
        ).to_csv(score_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(columns=["ticker", "date", "reason"]).to_csv(
            unavailable_path, index=False, encoding="utf-8-sig"
        )
        source_manifest = {
            "experiment_settings": {"name": "synthetic"},
            "model_spec": {"architecture": workflow.model_architecture},
            "training_objective": str(workflow_profile.training_objective),
            "training_sample_scope": str(workflow_profile.training_sample_scope),
        }
        model_record = build_file_manifest(source_model)
        score_record = build_file_manifest(score_path)
        score_record.update(
            {
                "columns": ["ticker", "date", "model_score"],
                "key_columns": ["ticker", "date"],
                "row_count": 1,
            }
        )
        unavailable_record = build_file_manifest(unavailable_path)
        unavailable_record.update(
            {
                "columns": ["ticker", "date", "reason"],
                "key_columns": ["ticker", "date"],
                "row_count": 0,
            }
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                    "model_architecture": workflow.model_architecture,
                    "experiment_profile": workflow.experiment_profile,
                    "shared_group_score_broadcast": True,
                    "model_identity": {
                        "model_checkpoint": model_record,
                        "model_information_cutoff": "2021-01-01",
                        **source_manifest,
                    },
                    "score_table": score_record,
                    "conservative_unscorable_events": unavailable_record,
                    "runtime_eligibility": {
                        "eligible": True,
                        "scope": "workflow_runtime",
                        "available_from": "2026-08-14",
                        "available_through": "2026-08-14",
                        "causal_information_contract": "same_day_and_earlier_ohlcv_only",
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        fake_contract = SimpleNamespace(
            model_information_cutoff="2021-01-01",
            manifest=source_manifest,
        )
        runtime_store.load_workflow_runtime_score_bundle.cache_clear()
        with (
            patch.object(runtime_store, "resolve_models_dir", return_value=str(models_root)),
            patch.object(runtime_store, "load_continuous_ranker_oos_contract", return_value=fake_contract),
            patch.object(
                runtime_store,
                "resolve_filter_artifact_paths",
                return_value=SimpleNamespace(model_path=source_model),
            ),
        ):
            available_rank = runtime_store.lookup_workflow_runtime_candidate_score(
                project_root=str(temp_root),
                ticker="2330",
                information_date="2026-08-14",
                high_len=201,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                model_architecture=workflow.model_architecture,
                experiment_profile=workflow.experiment_profile,
            )
            missing_rank = runtime_store.lookup_workflow_runtime_candidate_score(
                project_root=str(temp_root),
                ticker="2317",
                information_date="2026-08-14",
                high_len=201,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                model_architecture=workflow.model_architecture,
                experiment_profile=workflow.experiment_profile,
            )
            try:
                runtime_store.lookup_workflow_runtime_candidate_score(
                    project_root=str(temp_root),
                    ticker="2330",
                    information_date="2026-08-15",
                    high_len=201,
                    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                    model_architecture=workflow.model_architecture,
                    experiment_profile=workflow.experiment_profile,
                )
            except ValueError as exc:
                outside_coverage_rejected = "coverage不足" in str(exc)
            else:
                outside_coverage_rejected = False
        runtime_store.load_workflow_runtime_score_bundle.cache_clear()

    check_true(
        "workflow_runtime_daily_score_lookup_preserves_c44_unavailable_semantics",
        available_rank["available"] is True
                    and available_rank["score"] == 0.75
                    and available_rank["score_source"] == "canonical_runtime"
                    and missing_rank["available"] is False
                    and missing_rank["unavailable_reason"] == "missing_ticker_date_score"
                    and missing_rank["score_source"] == "canonical_runtime"
                    and outside_coverage_rejected,
    )
    blocked_mutations = []
    for blocked_action in ("run", "promote"):
        try:
            dispatch_runtime_integration_action(blocked_action)
        except RuntimeError as exc:
            blocked_mutations.append("historical compatibility" in str(exc))
        else:
            blocked_mutations.append(False)
    check_true(
        "runtime_integration_historical_mutations_fail_closed_before_producer_dispatch",
        blocked_mutations == [True, True],
    )

    check(
        "runtime_integration_decision_is_strict_three_state",
        ("GO", "BLOCKED", "NO_GO"),
        (
                    _overall_decision([{"status": "PASS"}]),
                    _overall_decision([{"status": "PASS"}, {"status": "BLOCKED"}]),
                    _overall_decision([{"status": "BLOCKED"}, {"status": "FAIL"}]),
                ),
    )

    certificate_status, certificate_evidence = _exact_certificate_status({
        "resource_aware_dl_selection_days": 5,
        "resource_aware_capital_utilization_days": 3,
        "resource_aware_max_dl_eligible_days": 5,
        "resource_aware_constrained_optimality_certified_days": 8,
        "resource_aware_max_dl_order_count_violation_days": 0,
        "resource_aware_preservation_violation_days": 0,
    })
    check_true(
        "runtime_integration_exact_certificate_uses_exact_active_days_not_max_dl_subset",
        certificate_status == "PASS"
                    and certificate_evidence["certificate_required_days"] == 8
                    and certificate_evidence["max_dl_eligible_days_diagnostic_only"] == 5
                    and certificate_evidence["certified_days"] == 8,
    )
    violation_status, _ = _exact_certificate_status({
        "resource_aware_dl_selection_days": 5,
        "resource_aware_capital_utilization_days": 3,
        "resource_aware_max_dl_eligible_days": 5,
        "resource_aware_constrained_optimality_certified_days": 8,
        "resource_aware_max_dl_order_count_violation_days": 1,
        "resource_aware_preservation_violation_days": 0,
    })
    missing_status, _ = _exact_certificate_status({
        "resource_aware_max_dl_eligible_days": 5,
        "resource_aware_constrained_optimality_certified_days": 5,
    })
    check(
        "runtime_integration_exact_certificate_violation_and_missing_evidence_are_distinct",
        ("FAIL", "BLOCKED"),
        (violation_status, missing_status),
    )

    gate_source = (
        Path(__file__).resolve().parents[2]
        / "services" / "research" / "runtime_integration_gate.py"
    ).read_text(encoding="utf-8")
    check_true(
        "runtime_integration_reuses_historical_outputs_without_mutating_runtime_default",
        "settings.reuse_output_roots" in gate_source
                    and "workflow.experiment_profile =" not in gate_source
                    and "run_runtime_integration_gate" in gate_source,
    )

    summary["candidate_profile"] = runtime["experiment_profile"]
    summary["current_anchor_profile"] = cfg.comparison_anchor_experiment_profile
    summary["promoted_workflow_profile"] = workflow.experiment_profile
    summary["gate_label"] = cfg.label
    return results, summary

