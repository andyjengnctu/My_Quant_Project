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
    add_check,
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

def validate_breakout_quality_single_seed_single_entry_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SINGLE_SEED_SINGLE_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    canonical_config_path = project_root / "config" / "breakout_quality.py"
    canonical_source = canonical_config_path.read_text(encoding="utf-8")
    canonical_app_path = project_root / "tools" / "filters" / "breakout_quality" / "application.py"
    canonical_app_source = canonical_app_path.read_text(encoding="utf-8")
    strategy_app_path = project_root / "apps" / "research.py"
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
        and "apps/research.py compare" in canonical_app_source,
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
    summary["strategy_compare_entry"] = "apps/research.py compare"
    return results, summary

def validate_strategy_compare_config_driven_app_contract_case(_base_params):
    case_id = "STRATEGY_COMPARE_CONFIG_DRIVEN_APP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "strategy_compare.py"
    app_path = project_root / "apps" / "research.py"
    model_app_path = project_root / "tools" / "filters" / "breakout_quality" / "application.py"
    orchestration_path = project_root / "filters" / "breakout_quality" / "strategy_comparison.py"
    preparation_path = project_root / "filters" / "breakout_quality" / "strategy_compare_preparation.py"
    engine_path = project_root / "filters" / "breakout_quality" / "strategy_compare_engine.py"
    contracts_path = project_root / "filters" / "breakout_quality" / "strategy_compare_contracts.py"
    sources_path = project_root / "filters" / "breakout_quality" / "strategy_compare_sources.py"
    reporting_path = project_root / "filters" / "breakout_quality" / "strategy_compare_reporting.py"
    diagnostics_path = project_root / "filters" / "breakout_quality" / "strategy_compare_diagnostics.py"
    replay_path = project_root / "filters" / "breakout_quality" / "strategy_compare_replay.py"
    export_service_path = project_root / "filters" / "breakout_quality" / "export_scores.py"
    param_service_path = project_root / "filters" / "breakout_quality" / "strategy_param_training.py"
    workflow_io_path = project_root / "filters" / "breakout_quality" / "workflow_io.py"
    optimizer_policy_path = project_root / "filters" / "breakout_quality" / "strategy_optimizer_policy.py"
    artifact_registry_path = project_root / "filters" / "breakout_quality" / "artifact_dependency_registry.py"
    portfolio_replay_service_path = project_root / "services" / "portfolio_replay.py"
    portfolio_replay_wrapper_path = project_root / "tools" / "portfolio_sim" / "simulation_runner.py"
    optimizer_raw_service_path = project_root / "services" / "optimizer" / "raw_cache.py"
    optimizer_trial_service_path = project_root / "services" / "optimizer" / "trial_inputs.py"
    optimizer_walk_forward_service_path = project_root / "services" / "optimizer" / "walk_forward.py"
    legacy_export_path = project_root / "tools" / "filters" / "breakout_quality" / "export_scores.py"
    legacy_param_path = project_root / "tools" / "filters" / "breakout_quality" / "strategy_dl_filter_param_adapt_gate.py"
    quick_gate_source = (project_root / "tools" / "local_regression" / "run_quick_gate.py").read_text(encoding="utf-8")
    audit_python_files = sorted((project_root / "tools" / "audit").rglob("*.py"))
    private_cross_audit_imports = []
    legacy_ranker_imports = []
    private_strategy_compare_engine_imports = []
    for audit_path in audit_python_files:
        audit_tree = read_source_ast(audit_path)
        for node in ast.walk(audit_tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            imported_names = [item.name for item in node.names]
            if node.module.startswith("tools.audit"):
                for imported_name in imported_names:
                    if imported_name.startswith("_"):
                        private_cross_audit_imports.append(
                            f"{audit_path.relative_to(project_root)}:{node.lineno}:{node.module}.{imported_name}"
                        )
            if node.module == "tools.filters.breakout_quality.train_continuous_ranker":
                legacy_ranker_imports.append(
                    f"{audit_path.relative_to(project_root)}:{node.lineno}:{','.join(imported_names)}"
                )
            if node.module == "filters.breakout_quality.strategy_compare_engine":
                for imported_name in imported_names:
                    if imported_name.startswith("_"):
                        private_strategy_compare_engine_imports.append(
                            f"{audit_path.relative_to(project_root)}:{node.lineno}:{imported_name}"
                        )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_modules_do_not_cross_import_other_audit_private_helpers",
        [],
        private_cross_audit_imports,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_modules_use_public_ranker_and_strategy_compare_owners",
        ([], []),
        (legacy_ranker_imports, private_strategy_compare_engine_imports),
    )

    from tools.audit import primitives as audit_primitives
    from tools.audit.breakout_quality import artifact_primitives, pit_primitives
    from tools.audit.breakout_quality import selection_replay_primitives, target_statistics
    from tools.audit.breakout_quality import continuous_target as continuous_target_audit
    from tools.audit.breakout_quality import no_time_continuous_target as no_time_target_audit
    from tools.audit.breakout_quality import selection_strategy_realization as selection_realization_audit
    from tools.audit.breakout_quality import candidate_counterfactual_execution as counterfactual_audit
    from tools.audit.breakout_quality import target_component_attribution as target_component_audit
    from tools.audit.breakout_quality import pit_fold_runtime_attribution as pit_fold_audit

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audits_share_one_statistics_implementation",
        (True, True, True),
        (
            continuous_target_audit._distribution_metrics is target_statistics.distribution_metrics,
            no_time_target_audit._distribution_metrics is target_statistics.distribution_metrics,
            continuous_target_audit._daily_rankability is no_time_target_audit._daily_rankability,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_and_pit_audits_share_identity_and_artifact_primitives",
        (True, True, True, True),
        (
            selection_realization_audit._attach_targets is selection_replay_primitives.attach_targets,
            counterfactual_audit._attach_targets is selection_replay_primitives.attach_targets,
            pit_fold_audit._candidate_pit_identity is pit_primitives.candidate_pit_identity,
            target_component_audit._sha256_file is audit_primitives.sha256_file
            and artifact_primitives.sha256_file is audit_primitives.sha256_file,
        ),
    )

    config_source = config_path.read_text(encoding="utf-8")
    app_source = app_path.read_text(encoding="utf-8")
    model_app_source = model_app_path.read_text(encoding="utf-8")
    orchestration_source = orchestration_path.read_text(encoding="utf-8")
    preparation_source = preparation_path.read_text(encoding="utf-8")
    strategy_compare_source = engine_path.read_text(encoding="utf-8")
    contracts_source = contracts_path.read_text(encoding="utf-8")
    sources_source = sources_path.read_text(encoding="utf-8")
    reporting_source = reporting_path.read_text(encoding="utf-8")
    diagnostics_source = diagnostics_path.read_text(encoding="utf-8")
    replay_source = replay_path.read_text(encoding="utf-8")
    param_service_source = param_service_path.read_text(encoding="utf-8")
    artifact_registry_source = artifact_registry_path.read_text(encoding="utf-8")
    portfolio_replay_service_source = portfolio_replay_service_path.read_text(encoding="utf-8")
    portfolio_replay_wrapper_source = portfolio_replay_wrapper_path.read_text(encoding="utf-8")
    optimizer_raw_service_source = optimizer_raw_service_path.read_text(encoding="utf-8")
    optimizer_trial_service_source = optimizer_trial_service_path.read_text(encoding="utf-8")
    optimizer_walk_forward_service_source = optimizer_walk_forward_service_path.read_text(encoding="utf-8")

    from config import strategy_compare as strategy_config
    from config.compatibility import strategy_compare_history as strategy_history
    from filters.breakout_quality.strategy_comparison import _pair_cache_required_files
    from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
    from core.strategy_comparison import strategy_comparison_fingerprint

    settings = strategy_config.get_strategy_comparison_settings()
    score_ranking_arm = next(
        arm
        for arm in settings.enabled_arms
        if arm.dl_enabled
        and strategy_config.get_strategy_comparison_settings(settings.profile_id).arms[arm.arm_id].dl_runtime_mode
        in {
            "resource-aware-continuous",
            "resource-aware-continuous-capital-preserving",
            "resource-aware-continuous-max-dl",
            "resource-aware-continuous-max-dl-feasible-ascent",
            "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
        }
    )
    score_ranking_cache_files = {
        path.name
        for path in _pair_cache_required_files(
            Path("synthetic-pair"),
            on_arm=score_ranking_arm,
        )
    }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_score_ranking_pairs_persist_execution_selector_trace_and_repair_search_certificate_sidecars_and_cache_requires_them",
        True,
        (
            "score_ranking_execution.csv" in score_ranking_cache_files
            and "score_ranking_selector_trace.csv" in score_ranking_cache_files
            and "score_ranking_repair_search_certificate.csv" in score_ranking_cache_files
            and "capture_execution_diagnostics=(" in orchestration_source
            and 'runtime_spec["comparison_mode"] == COMPARISON_MODE_SCORE_RANKING' in orchestration_source
            and 'output_dir / "score_ranking_execution.csv"' in strategy_compare_source
            and 'output_dir / "score_ranking_selector_trace.csv"' in strategy_compare_source
            and 'output_dir / "score_ranking_repair_search_certificate.csv"' in strategy_compare_source
        ),
    )
    configured_roos_builders = [
        source.builder
        for source in settings.parameter_sources.values()
        if source.builder is not None
        and source.builder.builder_type
        in {
            "binary_dl_min_roos_rolling",
            "selection_historical_p2",
            "selection_historical_full_roos",
        }
    ]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "all_roos_builders_use_training_policy_trials_single_source",
        True,
        bool(configured_roos_builders)
        and all(
            int(builder.options.get("trials_per_fold") or 0)
            == int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT)
            for builder in configured_roos_builders
        )
        and '"trials_per_fold": 200' not in config_source
        and '"baseline_trials_per_fold"' not in config_source,
    )
    from config.execution_policy import (
        DEFAULT_FIXED_RISK,
        DEFAULT_MAX_POSITION_CAP_PCT,
        DEFAULT_PORTFOLIO_MAX_POSITIONS,
        DEFAULT_PORTFOLIO_ROTATION,
    )
    from config.training_policy import OPTIMIZER_RANDOM_SEED_DEFAULT
    workflow_settings = strategy_config.get_breakout_quality_workflow_settings()
    builder_options = [
        dict(source.builder.options or {})
        for source in settings.parameter_sources.values()
        if source.builder is not None
    ]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_profile_membership_is_single_activation_source",
        True,
        all("enabled" not in raw for raw in strategy_config.STRATEGY_COMPARE_ARMS.values())
        and all("enabled" not in raw for raw in strategy_config.STRATEGY_COMPARE_CONTRASTS.values())
        and all("enabled" not in raw for raw in strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS.values())
        and all("enabled" not in raw for raw in strategy_history.HISTORICAL_STRATEGY_COMPARE_CONTRASTS.values())
        and all(
            {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings(profile_id).enabled_arms}
            == set(raw_profile["arm_ids"])
            and {
                contrast.contrast_id
                for contrast in strategy_config.get_strategy_comparison_settings(profile_id).enabled_contrasts
            } == set(raw_profile["contrast_ids"])
            for profile_id, raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.items()
        ),
    )
    active_arm_ids = {
        str(arm_id)
        for raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.values()
        for arm_id in tuple(raw_profile.get("arm_ids") or ())
    }
    active_contrast_ids = {
        str(contrast_id)
        for raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.values()
        for contrast_id in tuple(raw_profile.get("contrast_ids") or ())
    }
    active_param_ids = {
        str(strategy_config.STRATEGY_COMPARE_ARMS[arm_id]["param_source"])
        for arm_id in active_arm_ids
    }
    active_dl_ids = {
        str(strategy_config.STRATEGY_COMPARE_ARMS[arm_id].get("dl_id") or "")
        for arm_id in active_arm_ids
        if bool(strategy_config.STRATEGY_COMPARE_ARMS[arm_id].get("dl_enabled"))
    }
    active_dl_ids.discard("")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_active_and_historical_catalogs_are_physically_separated",
        True,
        set(strategy_config.STRATEGY_COMPARE_ARMS) == active_arm_ids
        and set(strategy_config.STRATEGY_COMPARE_CONTRASTS) == active_contrast_ids
        and set(strategy_config.STRATEGY_PARAM_SOURCES) == active_param_ids
        and set(strategy_config.STRATEGY_DL_SOURCES) == active_dl_ids
        and not set(strategy_config.STRATEGY_COMPARE_ARMS).intersection(
            strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS
        )
        and not set(strategy_config.STRATEGY_COMPARE_CONTRASTS).intersection(
            strategy_history.HISTORICAL_STRATEGY_COMPARE_CONTRASTS
        )
        and not set(strategy_config.STRATEGY_PARAM_SOURCES).intersection(
            strategy_history.HISTORICAL_STRATEGY_PARAM_SOURCES
        )
        and not set(strategy_config.STRATEGY_DL_SOURCES).intersection(
            strategy_history.HISTORICAL_STRATEGY_DL_SOURCES
        )
        and set(settings.arms)
        == active_arm_ids.union(strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS)
        and set(settings.contrasts)
        == active_contrast_ids.union(strategy_history.HISTORICAL_STRATEGY_COMPARE_CONTRASTS),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_execution_and_optimizer_defaults_have_config_ssot",
        True,
        int(workflow_settings.strategy_max_positions) == int(DEFAULT_PORTFOLIO_MAX_POSITIONS)
        and str(workflow_settings.strategy_rotation) == str(DEFAULT_PORTFOLIO_ROTATION)
        and float(workflow_settings.strategy_fixed_risk) == float(DEFAULT_FIXED_RISK)
        and float(workflow_settings.strategy_max_position_cap_pct) == float(DEFAULT_MAX_POSITION_CAP_PCT)
        and all(
            "fixed_risk" not in options
            or float(options["fixed_risk"]) == float(DEFAULT_FIXED_RISK)
            for options in builder_options
        )
        and all(
            "max_position_cap_pct" not in options
            or float(options["max_position_cap_pct"]) == float(DEFAULT_MAX_POSITION_CAP_PCT)
            for options in builder_options
        )
        and all(
            "optimizer_seed" not in options
            or int(options["optimizer_seed"]) == int(OPTIMIZER_RANDOM_SEED_DEFAULT)
            for options in builder_options
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_replay_service_boundary_has_canonical_implementation_and_tool_alias_only",
        True,
        portfolio_replay_service_path.is_file()
        and optimizer_raw_service_path.is_file()
        and optimizer_trial_service_path.is_file()
        and optimizer_walk_forward_service_path.is_file()
        and "from services.portfolio_replay import (" in replay_source
        and "from filters.breakout_quality.strategy_compare_replay import (" in strategy_compare_source
        and "from tools.portfolio_sim.simulation_runner import (" not in replay_source
        and "from tools." not in portfolio_replay_service_source
        and "import tools." not in portfolio_replay_service_source
        and "from tools." not in optimizer_raw_service_source
        and "from tools." not in optimizer_trial_service_source
        and "from tools." not in optimizer_walk_forward_service_source
        and "sys.modules[__name__] = _impl" in portfolio_replay_wrapper_source,
    )

    canonical_service_paths = (
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py",
        project_root / "services" / "breakout_quality" / "binary_point_in_time_scores.py",
        project_root / "services" / "breakout_quality" / "point_in_time_audit.py",
        project_root / "services" / "breakout_quality" / "continuous_ranker_pipeline.py",
        project_root / "services" / "optimizer" / "outer_rolling_oos.py",
        project_root / "services" / "optimizer" / "runtime.py",
        project_root / "services" / "optimizer" / "session_factory.py",
    )
    forbidden_reverse_imports: list[str] = []
    for root_name in ("services", "filters", "core"):
        for source_path in (project_root / root_name).rglob("*.py"):
            source_tree = read_source_ast(source_path)
            for node in ast.walk(source_tree):
                if isinstance(node, ast.ImportFrom) and str(node.module or "").startswith("tools"):
                    forbidden_reverse_imports.append(f"{source_path.relative_to(project_root)}:{node.lineno}:{node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if str(alias.name).startswith("tools"):
                            forbidden_reverse_imports.append(f"{source_path.relative_to(project_root)}:{node.lineno}:{alias.name}")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "formal_domain_and_services_do_not_reverse_depend_on_tools",
        True,
        all(path.is_file() for path in canonical_service_paths)
        and not forbidden_reverse_imports
        and "from services.breakout_quality.point_in_time_scores import (" in preparation_source
        and "from services.breakout_quality.point_in_time_audit import (" in preparation_source
        and "from services.breakout_quality.binary_point_in_time_scores import (" in param_service_source
        and "from services.optimizer.outer_rolling_oos import" in param_service_source
        and "from services.optimizer.runtime import" in param_service_source
        and "from services.optimizer.session_factory import (" in param_service_source,
    )

    import importlib
    canonical_train_module = importlib.import_module("services.breakout_quality.train")
    legacy_train_module = importlib.import_module("tools.filters.breakout_quality.train")
    canonical_optimizer_module = importlib.import_module("services.optimizer.outer_rolling_oos")
    legacy_optimizer_module = importlib.import_module("tools.optimizer.outer_rolling_oos")
    canonical_pit_module = importlib.import_module("services.breakout_quality.point_in_time_scores")
    legacy_pit_module = importlib.import_module("tools.filters.breakout_quality.build_point_in_time_scores")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "service_migration_preserves_legacy_module_aliases_and_project_root",
        True,
        canonical_train_module is legacy_train_module
        and canonical_optimizer_module is legacy_optimizer_module
        and canonical_pit_module is legacy_pit_module
        and Path(canonical_train_module.PROJECT_ROOT).resolve() == project_root.resolve(),
    )

    engine_tree = read_source_ast(engine_path)
    engine_defined_functions = {
        node.name for node in engine_tree.body if isinstance(node, ast.FunctionDef)
    }
    moved_strategy_compare_functions = {
        "_assert_controlled_param_pair",
        "_build_controlled_param_source_pair",
        "_load_param_source",
        "_resolve_params_path",
        "_scenario_summary",
        "_build_yearly_comparison",
        "_strategy_selection_diagnostics",
        "_run_scenario",
        "_resolve_comparison_period",
        "_load_reusable_no_filter_baseline",
        "run_standalone_baseline",
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_engine_delegates_sources_reporting_and_diagnostics_to_single_owner_modules",
        True,
        all(path.is_file() for path in (contracts_path, sources_path, reporting_path, diagnostics_path, replay_path))
        and not moved_strategy_compare_functions.intersection(engine_defined_functions)
        and "from filters.breakout_quality.strategy_compare_contracts import (" in strategy_compare_source
        and "from filters.breakout_quality.strategy_compare_sources import (" in strategy_compare_source
        and "from filters.breakout_quality.strategy_compare_reporting import (" in strategy_compare_source
        and "from filters.breakout_quality.strategy_compare_diagnostics import (" in strategy_compare_source
        and "from filters.breakout_quality.strategy_compare_replay import (" in strategy_compare_source
        and "def comparison_switch_spec(" in contracts_source
        and "def _build_controlled_param_source_pair(" in sources_source
        and "def _scenario_summary(" in reporting_source
        and "def _strategy_selection_diagnostics(" in diagnostics_source
        and "def _run_scenario(" in replay_source
        and "def run_standalone_baseline(" in replay_source
        and "from filters.breakout_quality.strategy_compare_reporting import (" in orchestration_source
        and "from filters.breakout_quality.strategy_compare_sources import (" in preparation_source
        and "from filters.breakout_quality.strategy_compare_sources import (" in param_service_source,
    )

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
            "get_strategy_comparison_profiles", "查看全部階段設定與工件狀態",
        ))
        and "C1" not in app_source and "TP1" not in app_source
        and '"strategy-compare"' not in model_app_source
        and "策略績效驗證" not in model_app_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_official_entry_is_in_quick_gate_help_registry",
        True,
        '([sys.executable, "apps/research.py", "compare", "--help"]' in quick_gate_source
        and '"apps/research.py",' in quick_gate_source.split("INLINE_CLI_TARGETS = {", 1)[1],
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
    pit_sources = [
        source for source in settings.dl_sources.values()
        if source.score_source == "selection_point_in_time"
    ]
    model_prepare_source = model_app_source.split(
        "def _prepare_strategy_compare_model_artifacts", 1
    )[1].split("def _interactive_model_research", 1)[0]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_compare_is_checkpoint_only_but_model_work_type_can_resume_train_missing_folds",
        True,
        bool(pit_sources)
        and all(source.threshold is None for source in pit_sources)
        and all(
            source.forward_scores_builder is not None
            and source.forward_scores_builder.builder_type
            == "selection_pit_from_existing_folds"
            for source in pit_sources
        )
        and "load_selection_point_in_time_ranking_contract" in preparation_source
        and "build_selection_point_in_time_scores" in preparation_source
        and "checkpoint_only=True" in preparation_source
        and "audit_selection_point_in_time_scores" in preparation_source
        and "build_selection_point_in_time_scores" in model_prepare_source
        and "resume=True" in model_prepare_source
        and "checkpoint_only=True" not in model_prepare_source
        and "缺少／不相容fold由模型訓練工作類型補訓" in model_prepare_source
        and "Strategy Compare不得因此訓練模型" in (
            project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
        ).read_text(encoding="utf-8")
        and any(
            source.artifact_contract is not None
            for source in settings.parameter_sources.values()
        ),
    )
    required_forward_oos_source_ids = {
        str(arm.dl_id)
        for arm in settings.enabled_arms
        if arm.dl_enabled
        and arm.dl_id
        and settings.dl_sources[str(arm.dl_id)].score_source
        == "continuous_ranker_oos"
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "model_work_type_prepares_all_configured_forward_oos_sources_without_strategy_compare_training",
        True,
        bool(required_forward_oos_source_ids)
        and "_strategy_compare_required_model_sources" in model_app_source
        and "SCORE_SOURCE_CONTINUOUS_RANKER_OOS" in model_prepare_source
        and "load_continuous_ranker_oos_contract" in model_prepare_source
        and "rebuild_forward_oos_scores_from_frozen_checkpoint" in model_prepare_source
        and '"train-continuous-ranker"' in model_prepare_source
        and "Forward-OOS policy：REUSE" in model_prepare_source
        and "Forward-OOS policy：模型工作類型修復" in model_prepare_source
        and "先重用frozen checkpoint" in model_prepare_source
        and "才重建完整模型工件" in model_prepare_source
        and "train-continuous-ranker" not in preparation_source
        and "模型訓練工作類型執行「準備策略比較所需模型工件」" in preparation_source,
    )
    selection_min_roos_source = settings.parameter_sources.get("selection_min_roos")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_historical_p2_parameter_source_has_formal_auto_builder",
        True,
        selection_min_roos_source is not None
        and selection_min_roos_source.builder is not None
        and selection_min_roos_source.builder.builder_type == "selection_historical_p2"
        and str(selection_min_roos_source.builder.options.get("parameter_set")) == "p2_history"
        and int(selection_min_roos_source.builder.options.get("trials_per_fold") or 0) >= 1
        and int(selection_min_roos_source.builder.options.get("train_window_months") or 0) >= 1
        and int(selection_min_roos_source.builder.options.get("oos_months") or 0) >= 1
        and "prepare_selection_historical_p2_params" in preparation_source
        and all(token in param_service_source for token in (
            "restore_selection_historical_p2_from_completed_strategy_compare",
            "completed_strategy_pair_exact_sha",
            "_build_min_roos_schedule_contract",
            "MIN_ROOS_SEARCH_FIELDS",
            "Selection Min ROOS缺少；自動建立／接續單階段rolling params",
        )),
    )
    selection_full_roos_source = settings.parameter_sources.get("selection_full_roos")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_historical_full_roos_has_independent_auto_builder_and_profile",
        True,
        selection_full_roos_source is not None
        and selection_full_roos_source.builder is not None
        and selection_full_roos_source.builder.builder_type == "selection_historical_full_roos"
        and str(selection_full_roos_source.builder.options.get("parameter_set")) == "p4_history"
        and "prepare_selection_historical_full_roos_params" in preparation_source
        and "FULL_ROOS_SEARCH_FIELDS" in param_service_source
        and "selection_full_roos_training" in param_service_source
        and {item["profile_id"] for item in strategy_config.get_strategy_comparison_profiles()}
        == set(strategy_config.STRATEGY_COMPARE_PROFILES)
        and all(
            {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings(profile_id).enabled_arms}
            == set(raw_profile["arm_ids"])
            and {
                item.contrast_id
                for item in strategy_config.get_strategy_comparison_settings(profile_id).enabled_contrasts
            } == set(raw_profile["contrast_ids"])
            and strategy_config.get_strategy_comparison_settings(profile_id).output_root
            == str(raw_profile["output_root"])
            and strategy_config.get_strategy_comparison_settings(profile_id).reuse_output_roots
            == tuple(raw_profile.get("reuse_output_roots", ()))
            for profile_id, raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.items()
        )
        and "_comparison_runs_roots" in orchestration_source,
    )

    display_alignment_groups = {}
    for profile_id, raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.items():
        group = str(raw_profile.get("display_alignment_group") or "").strip()
        if not group:
            continue
        display_alignment_groups.setdefault(group, []).append(
            tuple(
                arm.name
                for arm in strategy_config.get_strategy_comparison_settings(profile_id).enabled_arms
            )
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "configured_core_arm_display_alignment_groups_stay_aligned",
        True,
        bool(display_alignment_groups)
        and all(
            bool(group_names)
            and all(names == group_names[0] for names in group_names)
            and len(set(group_names[0])) == len(group_names[0])
            for group_names in display_alignment_groups.values()
        ),
    )

    robustness_path = (
        project_root / "filters" / "breakout_quality" / "strategy_multi_seed_robustness.py"
    )
    robustness_source = robustness_path.read_text(encoding="utf-8")
    robustness_settings = strategy_config.get_strategy_multi_seed_robustness_settings()
    robustness_profiles = strategy_config.get_strategy_multi_seed_robustness_profiles()
    selection_robustness = strategy_config.get_strategy_multi_seed_robustness_settings("selection_pit")
    robustness_profile = strategy_config.get_strategy_comparison_settings(
        robustness_settings.profile_id
    )
    robustness_fixed = tuple(
        arm for arm in robustness_profile.enabled_arms
        if arm.robustness_role == "fixed_baseline"
    )
    robustness_stochastic = tuple(
        arm for arm in robustness_profile.enabled_arms
        if arm.robustness_role == "stochastic"
    )
    reference_specs = dict(robustness_settings.romd_reference_baselines)
    reference_matches = {
        key: tuple(
            arm for arm in robustness_fixed
            if arm.param_source == spec["param_source"]
            and arm.rule_policy == spec["rule_policy"]
        )
        for key, spec in reference_specs.items()
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_robustness_is_strategy_compare_config_driven_without_second_arm_id_list",
        True,
        robustness_path.is_file()
        and {item["robustness_id"] for item in robustness_profiles} >= {"forward_oos", "selection_pit"}
        and selection_robustness.profile_id == "selection_pit"
        and robustness_settings.profile_id in strategy_config.STRATEGY_COMPARE_PROFILES
        and robustness_settings.seed_count >= 2
        and 1 <= robustness_settings.gpu_train_workers <= 2
        and robustness_settings.cpu_replay_workers >= 1
        and bool(robustness_fixed)
        and bool(robustness_stochastic)
        and all(not arm.dl_enabled for arm in robustness_fixed)
        and all(
            arm.dl_enabled
            and robustness_profile.dl_sources[str(arm.dl_id)].score_source
            == "continuous_ranker_oos"
            for arm in robustness_stochastic
        )
        and set(reference_matches) == {"min", "full"}
        and all(len(matches) == 1 for matches in reference_matches.values())
        and "romd_reference_baselines" in config_source
        and "robustness_role" in config_source
        and "MULTI_SEED_ROBUSTNESS_ARM_IDS" not in config_source
        and all(token not in robustness_source for token in ("\"C20\"", "\"C29\"", "\"MR-12B\"", "\"MR-13A\"", "\"Min ROOS\"", "\"Full ROOS\"")),
    )

    from filters.breakout_quality import strategy_multi_seed_robustness as robustness_module

    configured_seed_count = int(robustness_settings.seed_count)
    configured_generator_seed = int(robustness_settings.seed_generator_seed)
    seeds_a = robustness_module.resolve_multi_seed_values(
        seed_count=configured_seed_count, generator_seed=configured_generator_seed
    )
    seeds_b = robustness_module.resolve_multi_seed_values(
        seed_count=configured_seed_count, generator_seed=configured_generator_seed
    )
    seeds_c = robustness_module.resolve_multi_seed_values(
        seed_count=configured_seed_count, generator_seed=configured_generator_seed + 1
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_values_are_deterministically_generated_unique_and_not_best_seed_selection",
        True,
        seeds_a == seeds_b
        and seeds_a != seeds_c
        and len(seeds_a) == len(set(seeds_a)) == configured_seed_count
        and all(seed > 0 for seed in seeds_a)
        and "best_seed =" not in robustness_source.lower()
        and "selected_best_seed" not in robustness_source.lower(),
    )

    synthetic_period = {"start": "2001-01-01", "end": "2001-12-31"}
    deterministic_dataset_identity = {"dataset": "synthetic", "status": "READY", "sha256": "synthetic"}
    with patch.object(
        robustness_module, "_dataset_identity_snapshot", return_value=deterministic_dataset_identity
    ), patch.object(
        robustness_module, "get_strategy_multi_seed_robustness_settings", return_value=robustness_settings
    ):
        retention_contract_a = robustness_module.build_multi_seed_robustness_contract(
            robustness_id=robustness_settings.robustness_id,
            comparison_period=synthetic_period,
            artifact_identities={},
        )
    alternate_retention = replace(
        robustness_settings,
        keep_attribution_source=not robustness_settings.keep_attribution_source,
    )
    with patch.object(
        robustness_module, "_dataset_identity_snapshot", return_value=deterministic_dataset_identity
    ), patch.object(
        robustness_module, "get_strategy_multi_seed_robustness_settings", return_value=alternate_retention
    ):
        retention_contract_b = robustness_module.build_multi_seed_robustness_contract(
            robustness_id=alternate_retention.robustness_id,
            comparison_period=synthetic_period,
            artifact_identities={},
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_retention_policy_changes_do_not_change_scientific_fingerprint",
        (True, True),
        (
            retention_contract_a["fingerprint"] == retention_contract_b["fingerprint"],
            retention_contract_a["retention"]["keep_attribution_source"]
            != retention_contract_b["retention"]["keep_attribution_source"],
        ),
    )

    def _robustness_metrics(romd, total_return, win_rate):
        return {
            "total_return_pct": float(total_return),
            "max_drawdown_pct": 10.0,
            "return_over_max_drawdown": float(romd),
            "annual_return_pct": 12.0,
            "expected_value_r": 0.5,
            "payoff_ratio": 2.0,
            "avg_exposure_pct": 80.0,
            "trade_count": 100,
            "win_rate_pct": float(win_rate),
            "monthly_win_rate_pct": 60.0,
            "log_r_squared": 0.9,
            "direct_selection_r": 1.0,
        }

    min_reference_arm = reference_matches["min"][0]
    full_reference_arm = reference_matches["full"][0]
    fixed_results = {}
    for fixed_order, arm in enumerate(robustness_fixed, start=1):
        if arm.arm_id == full_reference_arm.arm_id:
            romd, total_return = 7.0, 120.0
        elif arm.arm_id == min_reference_arm.arm_id:
            romd, total_return = 6.0, 90.0
        else:
            romd, total_return = 5.0 + fixed_order / 10.0, 80.0 + fixed_order
        fixed_results[arm.arm_id] = {
            "metrics": _robustness_metrics(romd, total_return, 40.0),
            "yearly": [
                {"arm_id": arm.arm_id, "name": arm.name, "seed": None, "seed_order": None, "arm_order": fixed_order, "year": 2021, "return_pct": total_return / 10.0, "is_complete_year": True},
                {"arm_id": arm.arm_id, "name": arm.name, "seed": None, "seed_order": None, "arm_order": fixed_order, "year": 2022, "return_pct": total_return / 20.0, "is_complete_year": True},
            ],
        }
    synthetic_seed_rows = []
    synthetic_yearly_rows = []
    stochastic_expected_mean = {}
    for arm_order, arm in enumerate(robustness_stochastic, start=1):
        low = 5.0 + 3.0 * (arm_order - 1)
        high = low + 2.0
        stochastic_expected_mean[arm.arm_id] = (low + high) / 2.0
        for seed_order, (seed, romd) in enumerate(((101, low), (202, high)), start=1):
            synthetic_seed_rows.append({
                "arm_id": arm.arm_id,
                "name": arm.name,
                "seed": seed,
                "arm_order": arm_order,
                "seed_order": seed_order,
                **_robustness_metrics(romd, romd * 10.0, 50.0 + seed_order),
                "direct_selection_r": float(romd * 2.0),
            })
            for year, annual_return in ((2021, romd), (2022, romd + 1.0)):
                synthetic_yearly_rows.append({
                    "arm_id": arm.arm_id, "name": arm.name, "seed": seed,
                    "arm_order": arm_order, "seed_order": seed_order,
                    "year": year, "return_pct": annual_return, "is_complete_year": True,
                })
    synthetic_frame = pd.DataFrame(synthetic_seed_rows)
    synthetic_yearly_frame = pd.DataFrame(synthetic_yearly_rows)
    with patch.object(
        robustness_module, "build_source_data_inventory",
        return_value={"dataset": "synthetic-full"},
    ):
        synthetic_contract = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
            artifact_identities={
                "param:full_roos": {"sha256": "full"},
                "param:min_roos": {"sha256": "min"},
            },
        )
    synthetic_summary = robustness_module._robustness_summary(
        contract=synthetic_contract,
        fixed_results=fixed_results,
        seed_frame=synthetic_frame,
        seed_yearly_frame=synthetic_yearly_frame,
    )
    yearly_side_fixture = [{
        "year": 2021,
        "no_filter_return_pct": 1.25,
        "score_ranking_return_pct": 9.75,
        "is_full_year": True,
    }]
    fixed_yearly_side = robustness_module._normalize_yearly_rows(
        yearly_side_fixture, arm_id="fixed", name="Fixed", seed=None, seed_order=None, arm_order=1,
        result_side="no_filter",
    )
    stochastic_yearly_side = robustness_module._normalize_yearly_rows(
        yearly_side_fixture, arm_id="stochastic", name="Stochastic", seed=101, seed_order=1, arm_order=1,
    )
    mean_by_arm = {row["arm_id"]: row for row in synthetic_summary["mean_strategy_metrics"]}
    romd_by_arm = {row["arm_id"]: row for row in synthetic_summary["romd_statistics"]}
    stochastic_distribution_ok = all(
        math.isclose(
            mean_by_arm[arm.arm_id]["return_over_max_drawdown"],
            stochastic_expected_mean[arm.arm_id],
        )
        and math.isclose(mean_by_arm[arm.arm_id]["win_rate_pct"], 51.5)
        and math.isclose(romd_by_arm[arm.arm_id]["median"], stochastic_expected_mean[arm.arm_id])
        and math.isclose(
            romd_by_arm[arm.arm_id]["min"],
            stochastic_expected_mean[arm.arm_id] - 1.0,
        )
        and math.isclose(
            romd_by_arm[arm.arm_id]["max"],
            stochastic_expected_mean[arm.arm_id] + 1.0,
        )
        for arm in robustness_stochastic
    )
    distribution_compare = synthetic_summary["romd_distribution_comparison"]
    distribution_compare_ok = (
        distribution_compare is None
        if len(robustness_stochastic) != 2
        else isinstance(distribution_compare, dict)
    )
    same_seed_compare = synthetic_summary["romd_same_seed_comparison"]
    same_seed_compare_ok = (
        same_seed_compare is None
        if len(robustness_stochastic) != 2
        else (
            isinstance(same_seed_compare, dict)
            and same_seed_compare["n"] == 2
            and same_seed_compare["right_gt_left_count"] == 2
            and same_seed_compare["left_gt_right_count"] == 0
            and same_seed_compare["tie_count"] == 0
            and math.isclose(same_seed_compare["right_minus_left_mean"], 3.0)
            and math.isclose(same_seed_compare["right_minus_left_median"], 3.0)
        )
    )
    direct_same_seed_compare = synthetic_summary[
        "direct_selection_r_same_seed_comparison"
    ]
    direct_same_seed_compare_ok = (
        direct_same_seed_compare is None
        if len(robustness_stochastic) != 2
        else (
            isinstance(direct_same_seed_compare, dict)
            and direct_same_seed_compare["n"] == 2
            and direct_same_seed_compare["right_gt_left_count"] == 2
            and direct_same_seed_compare["left_gt_right_count"] == 0
            and direct_same_seed_compare["tie_count"] == 0
            and math.isclose(
                direct_same_seed_compare["right_minus_left_mean"], 6.0
            )
            and math.isclose(
                direct_same_seed_compare["right_minus_left_median"], 6.0
            )
        )
    )
    translation_diagnostic = synthetic_summary[
        "selection_r_to_strategy_same_seed_translation"
    ]
    translation_diagnostic_ok = (
        translation_diagnostic is None
        if len(robustness_stochastic) != 2
        else (
            isinstance(translation_diagnostic, dict)
            and translation_diagnostic["n"] == 2
            and translation_diagnostic["selection_r_positive_count"] == 2
            and translation_diagnostic["selection_r_positive_romd_positive_count"] == 2
            and translation_diagnostic["selection_r_positive_romd_nonpositive_count"] == 0
            and translation_diagnostic["sign_concordant_count"] == 2
            and translation_diagnostic["sign_discordant_count"] == 0
            and len(translation_diagnostic["seed_rows"]) == 2
            and [row["seed_index"] for row in translation_diagnostic["seed_rows"]] == [1, 2]
        )
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_report_uses_mean_for_all_strategy_metrics_and_full_romd_distribution_with_fixed_baselines",
        True,
        math.isclose(mean_by_arm[full_reference_arm.arm_id]["return_over_max_drawdown"], 7.0)
        and math.isclose(mean_by_arm[min_reference_arm.arm_id]["return_over_max_drawdown"], 6.0)
        and romd_by_arm[full_reference_arm.arm_id]["std"] is None
        and romd_by_arm[min_reference_arm.arm_id]["std"] is None
        and stochastic_distribution_ok
        and all(
            romd_by_arm[arm.arm_id]["beats_min_count"]
            == sum(
                value > 6.0
                for value in (
                    stochastic_expected_mean[arm.arm_id] - 1.0,
                    stochastic_expected_mean[arm.arm_id] + 1.0,
                )
            )
            and romd_by_arm[arm.arm_id]["beats_full_count"]
            == sum(
                value > 7.0
                for value in (
                    stochastic_expected_mean[arm.arm_id] - 1.0,
                    stochastic_expected_mean[arm.arm_id] + 1.0,
                )
            )
            for arm in robustness_stochastic
        )
        and synthetic_summary["contract"]["romd_reference_baselines"]["min"]["arm_id"]
        == min_reference_arm.arm_id
        and synthetic_summary["contract"]["romd_reference_baselines"]["full"]["arm_id"]
        == full_reference_arm.arm_id
        and distribution_compare_ok
        and same_seed_compare_ok
        and direct_same_seed_compare_ok
        and translation_diagnostic_ok
        and len(synthetic_summary["yearly_statistics"]) >= len(robustness_fixed) * 2 + len(robustness_stochastic) * 2
        and len(synthetic_summary["yearly_same_seed_comparison"]) == (2 if len(robustness_stochastic) == 2 else 0)
        and math.isclose(fixed_yearly_side[0]["return_pct"], 1.25)
        and math.isclose(stochastic_yearly_side[0]["return_pct"], 9.75)
        and robustness_source.count('result_side="no_filter"') >= 2
        and "_load_direct_selection_r(" in robustness_source
        and "active_trades_filename=runtime_spec[\"active_trades_filename\"]" in robustness_source,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_work_artifacts_are_isolated_and_retention_is_config_driven_with_optional_compact_attribution",
        True,
        all(token in robustness_source for token in (
            "--model-output-dir", "--research-output-dir",
            "keep_checkpoints", "keep_scores", "keep_replay_details", "keep_attribution_source",
            "seed_results.csv", "seed_yearly_returns.csv", "robustness_summary.json", "robustness_report.md",
            "attribution_source", "_write_compact_attribution_source", "REBUILD ATTRIBUTION",
            "ThreadPoolExecutor", "CPU strategy replay",
        ))
        and all(isinstance(value, bool) for value in (
            robustness_settings.keep_checkpoints,
            robustness_settings.keep_scores,
            robustness_settings.keep_replay_details,
            robustness_settings.keep_attribution_source,
        ))
        and "STRATEGY_COMPARE_ROBUSTNESS_KEEP_ATTRIBUTION_SOURCE" in config_source
        and "get_strategy_multi_seed_robustness_profiles" in app_source
        and "compare robustness" in app_source
        and "quiet=True" in robustness_source,
    )

    with tempfile.TemporaryDirectory() as tmp:
        compact_root = Path(tmp).resolve()
        pair_dir = compact_root / "pair"
        run_root = compact_root / "run"
        pair_dir.mkdir(parents=True)
        active_prefix = robustness_module._arm_runtime_spec(robustness_stochastic[0])["active_key"]
        for suffix in ("trades", "equity", "daily_capacity", "selected_buys", "execution"):
            pd.DataFrame([{"x": 1}, {"x": 2}]).to_csv(
                pair_dir / f"{active_prefix}_{suffix}.csv", index=False, encoding="utf-8-sig"
            )
        arm = robustness_stochastic[0]
        dl = robustness_profile.dl_sources[str(arm.dl_id)]
        runtime_spec = robustness_module._arm_runtime_spec(arm)
        attribution_dir = run_root / robustness_module.ATTRIBUTION_SOURCE_DIRNAME / f"{arm.arm_id}__seed_101"
        job = {
            "scientific_fingerprint": "synthetic-fingerprint",
            "robustness_id": robustness_settings.robustness_id,
            "profile_id": robustness_settings.profile_id,
            "arm_order": 1,
            "seed": 101,
            "seed_order": 1,
            "comparison_start": "2021-01-01",
            "comparison_end": "2022-12-31",
        }
        with patch.object(robustness_module, "PROJECT_ROOT", compact_root):
            compact_manifest = robustness_module._write_compact_attribution_source(
                pair_dir=pair_dir,
                destination_dir=attribution_dir,
                job=job,
                arm=arm,
                dl=dl,
                runtime_spec=runtime_spec,
            )
            pending_manifest = robustness_module._read_attribution_unit_manifest(
                run_root,
                arm_id=arm.arm_id,
                seed=101,
                expected_fingerprint="synthetic-fingerprint",
                require_verified=False,
            )
            preverify_ready = robustness_module._read_attribution_unit_manifest(
                run_root,
                arm_id=arm.arm_id,
                seed=101,
                expected_fingerprint="synthetic-fingerprint",
            )
            robustness_module._mark_attribution_unit_verified(
                run_root,
                arm_id=arm.arm_id,
                seed=101,
                expected_fingerprint="synthetic-fingerprint",
                source="synthetic_contract",
            )
            reloaded_manifest = robustness_module._read_attribution_unit_manifest(
                run_root,
                arm_id=arm.arm_id,
                seed=101,
                expected_fingerprint="synthetic-fingerprint",
            )
        compact_files_ok = (
            pending_manifest is not None
            and preverify_ready is None
            and reloaded_manifest is not None
            and dict(reloaded_manifest.get("scientific_observation_validation") or {}).get("status") == "VERIFIED"
            and compact_manifest["schema_version"] == robustness_module.ATTRIBUTION_SOURCE_SCHEMA_VERSION
            and set(compact_manifest["files"]) == {"trades", "equity", "daily_capacity", "selected_buys", "execution"}
            and all((attribution_dir / f"{role}.csv.gz").is_file() for role in compact_manifest["files"])
            and not any("orderable" in path.name or "score" in path.name for path in attribution_dir.iterdir())
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_compact_attribution_source_keeps_only_audit_required_active_replay_artifacts",
        True,
        compact_files_ok,
    )

    from filters.breakout_quality.strategy_compare_diagnostics import _flatten_entry_execution_rows
    synthetic_execution_frame = _flatten_entry_execution_rows([{
        "_event_type": "entry_execution",
        "execution_order": 7,
        "ticker": "2330",
        "type": "normal",
        "signal_date": "2024-01-02",
        "candidate_date": "2024-01-03",
        "trade_date": "2024-01-04",
        "limit_px": 100.0,
        "init_sl": 95.0,
        "candidate_qty": 1000,
        "chosen_qty": 800,
        "max_qty": None,
        "candidate_sizing_capital": 1_000_000.0,
        "sizing_equity": 1_000_000.0,
        "effective_entry_budget": 800_000.0,
        "available_cash_before": 800_000.0,
        "reserved_cost": 80_000.0,
        "params_obj": _base_params,
        "security_profile": None,
        "entry_filled": True,
        "filled_qty": 800,
        "actual_initial_risk_total_milli": 8_000_000,
        "entry_fill_price": 100.0,
    }])
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "entry_execution_diagnostics_capture_risk_utilization_and_cash_binding_without_changing_runtime",
        True,
        len(synthetic_execution_frame) == 1
        and int(synthetic_execution_frame.iloc[0]["execution_order"]) == 7
        and bool(synthetic_execution_frame.iloc[0]["entry_budget_cash_binding"])
        and "ENTRY_BUDGET_CASH" in str(synthetic_execution_frame.iloc[0]["binding_signature"])
        and pd.notna(synthetic_execution_frame.iloc[0]["risk_budget"])
        and pd.notna(synthetic_execution_frame.iloc[0]["actual_risk_utilization"]),
    )

    existing_observation = {key: 1.0 for _label, key, _unit in robustness_module.MEAN_METRICS}
    existing_observation.update({"selected_epoch": 2, "fold_count": None})
    rebuilt_observation = dict(existing_observation)
    rebuilt_observation["yearly"] = [
        {"year": 2024, "return_pct": 10.0, "is_complete_year": True}
    ]
    yearly_observation = pd.DataFrame(rebuilt_observation["yearly"])
    rebuild_identity_match = True
    try:
        robustness_module._validate_rebuilt_observation(
            existing_row=existing_observation,
            rebuilt=rebuilt_observation,
            existing_yearly=yearly_observation,
        )
    except RuntimeError:
        rebuild_identity_match = False
    drifted_observation = dict(rebuilt_observation)
    drifted_observation["selected_epoch"] = 3
    rebuild_identity_drift_blocked = False
    try:
        robustness_module._validate_rebuilt_observation(
            existing_row=existing_observation,
            rebuilt=drifted_observation,
            existing_yearly=yearly_observation,
        )
    except RuntimeError:
        rebuild_identity_drift_blocked = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_attribution_rebuild_reuses_scientific_observation_only_when_metrics_years_and_training_identity_match",
        (True, True),
        (rebuild_identity_match, rebuild_identity_drift_blocked),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_pipeline_allows_config_driven_two_gpu_trainers_without_changing_cpu_replay_policy",
        True,
        1 <= robustness_settings.gpu_train_workers <= 2
        and robustness_settings.cpu_replay_workers >= 1
        and "training_executor = ThreadPoolExecutor(" in robustness_source
        and "max_workers=int(cfg.gpu_train_workers)" in robustness_source
        and "replay_executor = ThreadPoolExecutor(" in robustness_source
        and "max_workers=int(cfg.cpu_replay_workers)" in robustness_source
        and "training_futures" in robustness_source
        and "_terminate_active_trainers" in robustness_source
        and "gpu_train_workers" in config_source
        and "STRATEGY_COMPARE_ROBUSTNESS_GPU_TRAIN_WORKERS" in config_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_selection_pit_preserves_partial_folds_and_exposes_fold_progress",
        True,
        'is_selection_pit = str(dl.score_source) == "selection_point_in_time"' in robustness_source
        and "PIT builder本身具備fold-level resume" in robustness_source
        and 'shutil.rmtree(model_dir / "folds"' not in robustness_source.split("def _train_one_unit", 1)[1].split("def _training_units", 1)[0]
        and "def _pit_saved_fold_progress" in robustness_source
        and "PIT saved folds=" in robustness_source
        and 'item["pit_saved_folds"]' in robustness_source
        and 'item["pit_expected_folds"]' in robustness_source
        and "if now >= next_print and (training_futures or replay_futures or ready_replays):" in robustness_source
        and 'tag = "[TRAIN]" if training_futures else "[REPLAY]"' in robustness_source,
    )

    from tools.filters.breakout_quality import train_continuous_ranker as ranker_train_module
    from filters.breakout_quality.ranking_score_store import (
        load_selection_point_in_time_score_table_from_path,
        lookup_continuous_ranker_oos_candidate_score,
    )

    with tempfile.TemporaryDirectory() as tmp:
        isolated_root = Path(tmp)
        isolated_model = isolated_root / "model"
        isolated_research = isolated_root / "research"
        isolated_args = ranker_train_module.parse_args([
            "--experiment-profile", "daily_universal_no_time_pairwise",
            "--model-output-dir", str(isolated_model),
            "--research-output-dir", str(isolated_research),
            "--seed", "123",
        ])
        isolated_paths, isolated_output = ranker_train_module._training_output_paths(
            isolated_args
        )
        isolated_score = isolated_root / "daily_scores.csv.gz"
        pd.DataFrame([{
            "ticker": "2330",
            "date": "2024-01-02",
            "group_index": 1,
            "model_score": 0.77,
        }]).to_csv(isolated_score, index=False, compression="gzip")
        isolated_lookup = lookup_continuous_ranker_oos_candidate_score(
            project_root=str(project_root),
            ticker="2330",
            signal_date="2024-01-02",
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_pairwise",
            score_path_override=str(isolated_score),
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_training_and_score_lookup_support_isolated_paths_without_touching_canonical_model",
        True,
        isolated_paths.model_dir == isolated_model.resolve()
        and isolated_output == isolated_research.resolve()
        and math.isclose(float(isolated_lookup["score"]), 0.77)
        and isolated_lookup["available"]
        and isolated_lookup["continuous_target_id"] == "daily_opportunity_no_time_r_v1",
    )

    with tempfile.TemporaryDirectory() as tmp:
        pit_root = Path(tmp)
        pit_score = pit_root / "selection_point_in_time_scores.csv"
        pit_manifest = pit_root / "selection_point_in_time_manifest.json"

        def _write_pit_score(score_value: float) -> None:
            pd.DataFrame([{
                "ticker": "2330",
                "date": "2020-01-02",
                "group_index": 1,
                "breakout_quality_score": score_value,
                "fold_id": "F1",
                "model_information_cutoff": "2019-12-31",
            }]).to_csv(pit_score, index=False)
            pit_manifest.write_text(json.dumps({
                "score_period": {"start": "2020-01-02", "end": "2020-01-02"},
                "coverage": {"scored_group_count": 1},
            }), encoding="utf-8")

        _write_pit_score(0.70)
        pit_first = load_selection_point_in_time_score_table_from_path(
            str(pit_score), manifest_path=str(pit_manifest)
        )
        pit_second = load_selection_point_in_time_score_table_from_path(
            str(pit_score), manifest_path=str(pit_manifest)
        )
        _write_pit_score(0.812345)
        pit_rebuilt = load_selection_point_in_time_score_table_from_path(
            str(pit_score), manifest_path=str(pit_manifest)
        )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "isolated_selection_pit_lookup_caches_same_file_revision_and_invalidates_rebuild",
        True,
        pit_first is pit_second
        and pit_rebuilt is not pit_first
        and math.isclose(
            float(pit_rebuilt.loc[("2330", "2020-01-02"), "breakout_quality_score"]),
            0.812345,
        ),
    )

    from filters.breakout_quality.strategy_compare_engine import (
        _resolve_continuous_score_override_period,
    )

    isolated_period_table = pd.DataFrame(
        {"model_score": [0.5]},
        index=pd.MultiIndex.from_tuples(
            [("2330", "2021-01-04")], names=["ticker", "date"]
        ),
    )
    isolated_period_table.attrs["available_from"] = "2021-01-04"
    isolated_period_table.attrs["available_through"] = "2026-03-02"
    isolated_execution_start, isolated_available_from, isolated_available_through = (
        _resolve_continuous_score_override_period(
            isolated_period_table, execution_start_override="2021-01-01"
        )
    )
    late_execution_rejected = False
    try:
        _resolve_continuous_score_override_period(
            isolated_period_table, execution_start_override="2021-01-05"
        )
    except ValueError:
        late_execution_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_isolated_score_override_separates_calendar_execution_start_from_first_trading_score_date",
        True,
        isolated_execution_start == "2021-01-01"
        and isolated_available_from == "2021-01-04"
        and isolated_available_through == "2026-03-02"
        and late_execution_rejected
        and '"score_execution_start": artifacts["score_execution_start"]' in robustness_source
        and 'continuous_score_execution_start_override=' in robustness_source
        and 'str(job["score_execution_start"])' in robustness_source
        and "outer_oos_policy" in robustness_source,
    )
    engine_source = (
        project_root / "filters" / "breakout_quality" / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_override_coverage_metadata_keeps_execution_start_distinct_from_first_score",
        True,
        '"required_start": continuous_score_override["execution_start"]' in engine_source
        and '"first_scored_event": continuous_score_override["available_from"]' in engine_source,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_multi_seed_uses_isolated_period_scoped_pit_scores_and_dynamic_menu_profile",
        True,
        '"selection_pit"' in config_source
        and "--point-in-time-dir-override" in robustness_source
        and "--score-start-date" in robustness_source
        and "--score-end-date" in robustness_source
        and "selection_pit_score_path_override" in engine_source
        and "selection_pit_manifest_path_override" in engine_source
        and "get_strategy_multi_seed_robustness_profiles" in app_source
        and "_pit_fold_count_for_period" in robustness_source,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_multi_seed_skips_non_runtime_future_target_join_during_per_seed_replay",
        True,
        "capture_selection_target_diagnostics=True" in engine_source
        and "and bool(capture_selection_target_diagnostics)" in engine_source
        and "capture_selection_target_diagnostics=False" in robustness_source
        and "daily-universal profile" in robustness_source,
    )

    from core.strategy_comparison import StrategyPreparationAction, StrategyPreparationPlan
    from filters.breakout_quality import strategy_compare_preparation as preparation_module

    dl_blocked = StrategyPreparationAction(
        action_id="dl:synthetic:model", artifact_key="dl:synthetic:model",
        action="BLOCKED", builder_type=None, description="canonical DL intentionally absent", path="models/dl.pt",
    )
    fake_settings = robustness_profile
    configured_required_sources = sorted({
        arm.param_source for arm in (*robustness_fixed, *robustness_stochastic)
    })
    parameter_only_prepare_ok = False
    if len(configured_required_sources) >= 2:
        build_source, reuse_source = configured_required_sources[:2]
        first_plan = StrategyPreparationPlan(
            overall_status="BLOCKED",
            actions=(
                StrategyPreparationAction(
                    action_id=f"param:{build_source}", artifact_key=f"param:{build_source}",
                    action="BUILD", builder_type="synthetic_builder", description="build", path="models/build.json",
                ),
                StrategyPreparationAction(
                    action_id=f"param:{reuse_source}", artifact_key=f"param:{reuse_source}",
                    action="REUSE", builder_type=None, description="reuse", path="models/reuse.json",
                ),
                dl_blocked,
            ),
        )
        second_plan = StrategyPreparationPlan(
            overall_status="BLOCKED",
            actions=(
                StrategyPreparationAction(
                    action_id=f"param:{build_source}", artifact_key=f"param:{build_source}",
                    action="REUSE", builder_type=None, description="reuse", path="models/build.json",
                ),
                StrategyPreparationAction(
                    action_id=f"param:{reuse_source}", artifact_key=f"param:{reuse_source}",
                    action="REUSE", builder_type=None, description="reuse", path="models/reuse.json",
                ),
                dl_blocked,
            ),
        )
        prep_calls = []
        with patch.object(
            preparation_module, "_execute_preparation_action",
            side_effect=lambda **kwargs: prep_calls.append(kwargs["action"].artifact_key),
        ):
            prepared = preparation_module.prepare_strategy_parameter_artifacts(
                settings=fake_settings,
                status={"preparation_plan": first_plan},
                required_source_ids=(build_source, reuse_source),
                status_refresher=lambda: {"preparation_plan": second_plan},
            )
        parameter_only_prepare_ok = (
            prep_calls == [f"param:{build_source}"]
            and prepared["preparation_plan"] is second_plan
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_parameter_preparation_ignores_normal_dl_blockers_and_reuses_canonical_param_builder",
        True,
        parameter_only_prepare_ok
        and "prepare_strategy_parameter_artifacts" in robustness_source
        and "請先由Strategy Compare正式前置建立" not in robustness_source,
    )

    with tempfile.TemporaryDirectory() as tmp:
        period_root = Path(tmp)
        (period_root / "dataset_summary.json").write_text(
            json.dumps({"dataset": robustness_profile.dataset, "source_data_date_range": {"end": "2026-03-02"}}),
            encoding="utf-8",
        )
        stale_status = {
            "comparison_period": {"start": "2021-01-01", "end": "2025-12-22"}
        }
        with patch.object(
            robustness_module, "resolve_filter_output_dir", return_value=period_root
        ), patch.object(
            robustness_module, "resolve_breakout_quality_outer_policy",
            return_value={"oos_start_date": "2021-01-01", "effective_oos_end_date": "2026-03-02"},
        ):
            derived_start, derived_end = robustness_module._comparison_period_from_upstream(
                robustness_profile, robustness_stochastic, stale_status
            )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_forward_period_is_derived_from_dataset_policy_not_current_canonical_score_tail",
        True,
        derived_start == "2021-01-01"
        and derived_end == "2026-03-02",
    )

    dataset_profile_mismatch_rejected = False
    with tempfile.TemporaryDirectory() as tmp:
        period_root = Path(tmp)
        (period_root / "dataset_summary.json").write_text(
            json.dumps({"dataset": "wrong-profile", "source_data_date_range": {"end": "2026-03-02"}}),
            encoding="utf-8",
        )
        with patch.object(
            robustness_module, "resolve_filter_output_dir", return_value=period_root
        ):
            try:
                robustness_module._comparison_period_from_upstream(
                    robustness_profile, robustness_stochastic, {}
                )
            except RuntimeError:
                dataset_profile_mismatch_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_forward_period_rejects_dataset_profile_mismatch",
        True,
        dataset_profile_mismatch_rejected,
    )

    valid_resume_rows = []
    resume_seeds = (101, 202)
    for arm_order, arm in enumerate(robustness_stochastic, start=1):
        for seed_order, seed in enumerate(resume_seeds, start=1):
            row = {
                "arm_id": arm.arm_id, "seed": seed, "arm_order": arm_order,
                "seed_order": seed_order,
            }
            row.update(_robustness_metrics(6.0 + arm_order, 100.0, 50.0))
            valid_resume_rows.append(row)
    valid_resume_frame = pd.DataFrame(valid_resume_rows)
    robustness_module._validate_seed_results_frame(
        valid_resume_frame, stochastic_arms=robustness_stochastic, seeds=resume_seeds
    )
    duplicate_rejected = False
    try:
        robustness_module._validate_seed_results_frame(
            pd.concat([valid_resume_frame, valid_resume_frame.iloc[[0]]], ignore_index=True),
            stochastic_arms=robustness_stochastic, seeds=resume_seeds,
        )
    except ValueError:
        duplicate_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_resume_rejects_duplicate_or_foreign_seed_result_observations",
        True,
        duplicate_rejected
        and '"failed_stage": "resume_preflight_or_fixed_baseline"' in robustness_source
        and '"failed_stage": "summary_or_report_finalization"' in robustness_source,
    )

    confirm_index = robustness_source.find("按 Enter 執行全部前置、訓練與回放")
    parameter_prepare_index = robustness_source.find("status = prepare_strategy_parameter_artifacts(")
    run_dir_index = robustness_source.find("run_root.mkdir(parents=True, exist_ok=True)")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_shows_one_plan_confirmation_before_parameter_build_and_run_artifact_creation",
        True,
        confirm_index >= 0
        and parameter_prepare_index > confirm_index
        and run_dir_index > parameter_prepare_index
        and robustness_source.count("按 Enter 執行全部前置、訓練與回放") == 1,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_replay_failure_is_resumable_and_stops_overlapped_training_process",
        True,
        '"status": "FAILED"' in robustness_source
        and '"resumable": True' in robustness_source
        and "proc.terminate()" in robustness_source
        and "proc.kill()" in robustness_source
        and "subprocess.TimeoutExpired" in robustness_source,
    )

    with patch.object(
        robustness_module, "build_source_data_inventory",
        return_value={"dataset": "synthetic-full", "inventory": "same"},
    ):
        fingerprint_base = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
            artifact_identities={"param:full_roos": {"sha256": "a"}, "param:min_roos": {"sha256": "b"}},
        )["fingerprint"]
        fingerprint_period = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2026-01-31"},
            artifact_identities={"param:full_roos": {"sha256": "a"}, "param:min_roos": {"sha256": "b"}},
        )["fingerprint"]
        fingerprint_param = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
            artifact_identities={"param:full_roos": {"sha256": "changed"}, "param:min_roos": {"sha256": "b"}},
        )["fingerprint"]
        with patch.object(
            robustness_module, "BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE",
            float(robustness_module.BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE) * 2.0,
        ):
            fingerprint_training = robustness_module.build_multi_seed_robustness_contract(
                comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
                artifact_identities={"param:full_roos": {"sha256": "a"}, "param:min_roos": {"sha256": "b"}},
            )["fingerprint"]
        display_only_settings = replace(
            robustness_settings,
            console_mode=("verbose" if robustness_settings.console_mode == "compact" else "compact"),
            progress_interval_seconds=float(robustness_settings.progress_interval_seconds) + 7.0,
            yearly_report=not bool(robustness_settings.yearly_report),
            keep_replay_details=not bool(robustness_settings.keep_replay_details),
        )
        with patch.object(
            robustness_module, "get_strategy_multi_seed_robustness_settings",
            return_value=display_only_settings,
        ):
            fingerprint_display_only = robustness_module.build_multi_seed_robustness_contract(
                comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
                artifact_identities={"param:full_roos": {"sha256": "a"}, "param:min_roos": {"sha256": "b"}},
            )["fingerprint"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_fingerprint_changes_with_oos_period_param_identity_and_effective_training_defaults",
        True,
        len({fingerprint_base, fingerprint_period, fingerprint_param, fingerprint_training}) == 4
        and fingerprint_display_only == fingerprint_base,
    )

    from filters.breakout_quality import strategy_param_training as strategy_param_training_module

    with tempfile.TemporaryDirectory() as tmp:
        selection_full_root = Path(tmp)
        optimizer_calls = []

        def _fake_selection_full_outer_rolling(**kwargs):
            optimizer_calls.append(kwargs)
            destination = Path(kwargs["paramset_models_dir"])
            destination.mkdir(parents=True, exist_ok=True)
            params = {
                "high_len": 201,
                "atr_len": 14,
                "atr_buy_tol": 1.5,
                "atr_times_init": 2.0,
                "atr_times_trail": 3.0,
                "use_breakout_quality_filter": False,
                "use_breakout_quality_ranking": False,
                "use_history_threshold": False,
                "tp_percent": 0.0,
            }
            payload = {
                "meta": {
                    "first_oos_date": "2014-01-01",
                    "last_oos_date": "2020-12-01",
                    "train_window_months": 120,
                    "oos_horizon_months": 12,
                    "trials_per_fold": int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
                },
                "summary": {"folds": 7},
                "params_ensemble_by_effective_date": {
                    f"{year}-01-01": [{"params": dict(params)}]
                    for year in range(2014, 2021)
                },
            }
            (destination / "roos_base_best.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            return 0

        with patch.object(
            strategy_param_training_module,
            "build_source_data_inventory",
            return_value={"dataset": "synthetic-full"},
        ), patch.object(
            strategy_param_training_module,
            "build_rolling_base_policy",
            return_value={"model_mode": "oos"},
        ), patch.object(
            strategy_param_training_module,
            "get_dataset_dir",
            return_value=str(selection_full_root / "data"),
        ), patch.object(
            strategy_param_training_module,
            "run_outer_rolling_oos",
            side_effect=_fake_selection_full_outer_rolling,
        ):
            selection_full_kwargs = {
                "project_root": selection_full_root,
                "dataset": "full",
                "param_policy": "base-finalist-best",
                "trials_per_fold": int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT),
                "max_positions": 10,
                "rotation": "off",
                "fixed_risk": 0.01,
                "max_position_cap_pct": 0.30,
                "optimizer_seed": 42,
                "quiet": True,
                "first_oos_date": "2014-01-01",
                "last_oos_date": "2020-12-31",
                "train_window_months": 120,
                "oos_months": 12,
            }
            selection_full_first = strategy_param_training_module.prepare_selection_historical_full_roos_params(
                **selection_full_kwargs
            )
            selection_full_second = strategy_param_training_module.prepare_selection_historical_full_roos_params(
                **selection_full_kwargs
            )
            selection_full_payload = json.loads(
                Path(selection_full_second["params_path"]).read_text(encoding="utf-8")
            )
    full_search_fields = set(
        selection_full_payload["breakout_quality_param_adaptation"]["search_fields"]
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_historical_full_roos_builder_runs_once_reuses_and_excludes_fixed_fields",
        True,
        len(optimizer_calls) == 1
        and not bool(selection_full_first["summary"]["optimizer_search_reused"])
        and bool(selection_full_second["summary"]["optimizer_search_reused"])
        and int(selection_full_second["summary"]["folds"]) == 7
        and selection_full_payload["breakout_quality_param_adaptation"]["mode"]
        == "selection_full_roos_training"
        and selection_full_payload["breakout_quality_param_adaptation"]["parameter_set"]
        == "P4_HISTORY"
        and {"high_len", "atr_len", "atr_buy_tol", "atr_times_init", "atr_times_trail"}
        <= full_search_fields
        and all(
            field not in full_search_fields
            for field in (
                "use_breakout_buy",
                "use_breakout_quality_filter",
                "use_breakout_quality_ranking",
                "use_history_threshold",
                "min_history_trades",
                "min_history_ev",
                "min_history_win_rate",
            )
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "completed_pair_cache_can_replace_vanished_historical_pit_only_for_reuse",
        True,
        all(token in orchestration_source for token in (
            "archived_completed_pair",
            "_find_reusable_pair_with_archived_source",
            "stored_param_sha != current_param_sha",
            "_archived_pair_source_is_self_contained",
            "settings.dl_sources[dl_id].score_source != SCORE_SOURCE_SELECTION_POINT_IN_TIME",
            "dependent_arms = tuple(",
            "本次使用此DL的arms全部重用identity一致的completed pair",
        )),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_builds_parameter_identity_first_and_replans_before_historical_pit_rebuild",
        True,
        all(token in preparation_source for token in (
            "status_refresher",
            "execution_priority=10",
            "selected.next_runnable_action",
            "current = status_refresher()",
        )),
    )

    from filters.breakout_quality.artifact_dependency_registry import (
        ARTIFACT_CONTINUOUS_TARGET,
        ARTIFACT_DATASET_CORE,
        ARTIFACT_FORWARD_SCORE,
        ARTIFACT_MODEL_CHECKPOINT,
        ARTIFACT_SELECTION_PIT_AUDIT,
        ARTIFACT_SELECTION_PIT_SCORE,
        required_upstream_artifact_types,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "artifact_dependency_registry_owns_model_truth_and_score_chain",
        True,
        artifact_registry_path.is_file()
        and all(token in artifact_registry_source for token in (
            "ARTIFACT_DEPENDENCY_REGISTRY",
            "collect_model_upstream_readiness",
            "dependency_types_for",
        ))
        and set((
            ARTIFACT_DATASET_CORE, ARTIFACT_CONTINUOUS_TARGET, ARTIFACT_MODEL_CHECKPOINT,
            ARTIFACT_FORWARD_SCORE, ARTIFACT_SELECTION_PIT_SCORE, ARTIFACT_SELECTION_PIT_AUDIT,
        )).issubset(set(__import__(
            "filters.breakout_quality.artifact_dependency_registry",
            fromlist=["ARTIFACT_DEPENDENCY_REGISTRY"],
        ).ARTIFACT_DEPENDENCY_REGISTRY))
        and required_upstream_artifact_types(
            "strategy_aligned_no_time_all_event_pairwise"
        ) == (ARTIFACT_DATASET_CORE, ARTIFACT_CONTINUOUS_TARGET)
        and required_upstream_artifact_types(
            "daily_universal_no_time_pairwise"
        ) == (ARTIFACT_DATASET_CORE,),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_and_model_work_share_artifact_upstream_readiness_registry",
        True,
        "collect_model_upstream_readiness" in preparation_source
        and "collect_model_upstream_readiness" in model_app_source
        and "model-upstream:" in preparation_source
        and "source_upstream_dependencies" in preparation_source,
    )

    from filters.breakout_quality.strategy_compare_preparation import (
        _validate_expected_artifact_contract,
        model_upstream_prerequisite_blockers,
    )
    with tempfile.TemporaryDirectory() as missing_upstream_temp:
        missing_root = Path(missing_upstream_temp)
        event_blockers = model_upstream_prerequisite_blockers(
            missing_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="strategy_aligned_no_time_all_event_pairwise",
            dataset="full",
        )
        daily_blockers = model_upstream_prerequisite_blockers(
            missing_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_pairwise",
            dataset="full",
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_checkpoint_rebuild_blocks_before_runtime_when_model_upstream_is_missing",
        True,
        len(event_blockers) == 2
        and any("canonical Dataset" in item for item in event_blockers)
        and any("Continuous Target" in item for item in event_blockers)
        and len(daily_blockers) == 1
        and any("canonical Dataset" in item for item in daily_blockers)
        and not any("market-set" in item for item in daily_blockers)
        and "Strategy Compare不得建立Dataset／Label／Target" in preparation_source,
    )
    contract_example = {
        "breakout_quality_param_adaptation": {
            "mode": "min_roos_training",
            "parameter_set": "P2_HISTORY",
            "search_fields": [
                "high_len",
                "atr_len",
                "atr_buy_tol",
                "atr_times_init",
                "atr_times_trail",
            ],
            "fixed_rule_contract": "all_rule_filters_off",
            "training_dl_enabled": False,
        },
        "unrelated": 123,
    }
    expected_contract = {
        "breakout_quality_param_adaptation": {
            "parameter_set": "P2_HISTORY",
            "training_dl_enabled": False,
        }
    }
    bad_contract = {
        "breakout_quality_param_adaptation": {
            "parameter_set": "P3",
            "training_dl_enabled": False,
        }
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_artifact_subset_contract_accepts_expected_identity_and_rejects_mismatch",
        True,
        _validate_expected_artifact_contract(contract_example, expected_contract) is None
        and _validate_expected_artifact_contract(bad_contract, expected_contract) is not None,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_has_ready_preparable_blocked_and_single_confirmation_contract",
        True,
        all(token in preparation_source for token in (
            "StrategyPreparationPlan.from_actions(actions)",
            'failure_prefix="策略比較前置"',
            'f"{failure_prefix}失敗:',
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

    portfolio_source = (
        project_root / "core" / "portfolio_entry_selection_max_dl.py"
    ).read_text(encoding="utf-8")
    max_dl_source = portfolio_source.split("def _max_dl_execution_order", 1)[1]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_feasible_ascent_resource_floor_is_same_param_baseline_not_min_roos_specific",
        True,
        "same_param_exact_resource_baseline" in strategy_compare_source
        and "canonical_same_param_exact_cash_cap_baseline" in strategy_compare_source
        and "min_roos_exact_resource_baseline" not in strategy_compare_source
        and "canonical_min_roos_exact_cash_cap_baseline" not in strategy_compare_source
        and "同參數DL-off baseline" in max_dl_source
        and "seed不符合Min ROOS資源契約" not in max_dl_source
        and settings.arms["C30"].param_source == "full_roos"
        and settings.arms["C31"].param_source == "full_roos"
        and settings.arms["C30"].dl_runtime_mode
        == settings.arms["C31"].dl_runtime_mode
        == "resource-aware-continuous-max-dl-feasible-ascent",
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
        and {"C7", "C8", "C9", "C10", "C11", "C12", "C14", "C15", "C16", "C17", "C18"}.issubset(set(settings.arms))
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
    mismatched_arms["C20"] = _replace_strategy_arm(
        mismatched_arms["C20"], param_source="min_dl_tp1_roos", dl_id="CONT12B"
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
        "C16": "Min ROOS: All-event Continuous capital-preserving",
        "C17": "Min ROOS: All-event Continuous max-DL constrained basket",
        "C18": "Min ROOS: All-event Continuous max-DL feasible-ascent",
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "min_roos_display_names_follow_training_identity_then_runtime_suffix_contract",
        True,
        all(
            str(settings.arms[arm_id].name) == name
            for arm_id, name in expected_display_names.items()
        ),
    )

    from core.exact_accounting import build_buy_ledger_from_price
    from core.portfolio_entries import (
        _simulate_reserved_candidate_order,
        reorder_candidates_for_resource_aware_quality,
        select_resource_aware_action_candidates,
    )
    from core.portfolio_entry_selection_max_dl import (
        build_max_dl_repair_mechanism_diagnostic,
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
    resource_order, resource_diag = reorder_candidates_for_resource_aware_quality(
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
    slot_order, slot_diag = reorder_candidates_for_resource_aware_quality(
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
    greedy_order, greedy_diag = reorder_candidates_for_resource_aware_quality(
        greedy_rows,
        available_cash=548_426.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    basket_order, basket_diag = reorder_candidates_for_resource_aware_quality(
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
    continuous_order, continuous_diag = reorder_candidates_for_resource_aware_quality(
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
    continuous_slot_order, continuous_slot_diag = reorder_candidates_for_resource_aware_quality(
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

    capital_preserving_slot_rows = [
        _resource_candidate_fixed(
            "R1", 1000.0, 70, 0.10,
            "resource-aware-continuous-capital-preserving",
        ),
        _resource_candidate_fixed(
            "Q1", 1000.0, 80, 0.95,
            "resource-aware-continuous-capital-preserving",
        ),
    ]
    capital_preserving_slot_order, capital_preserving_slot_diag = (
        reorder_candidates_for_resource_aware_quality(
            capital_preserving_slot_rows,
            available_cash=180_000.0,
            sizing_equity=2_000_000.0,
            pre_market_occupied=9,
            max_positions=10,
            params=resource_params,
        )
    )
    capital_preserving_slot_selected = _simulate_reserved_candidate_order(
        capital_preserving_slot_order,
        available_cash=180_000.0,
        sizing_equity=2_000_000.0,
        free_slots=1,
        params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capital_preserving_continuous_can_improve_slot_binding_day_without_reducing_count_or_reserved_capital",
        True,
        capital_preserving_slot_diag["mode"] == "dl-selection"
        and capital_preserving_slot_diag["changed"]
        and capital_preserving_slot_diag["resource_preservation_required"]
        and capital_preserving_slot_diag["selected_count_preserved"]
        and capital_preserving_slot_diag["reserved_capital_preserved"]
        and capital_preserving_slot_diag["selected_count"]
        >= capital_preserving_slot_diag["baseline_selected_count"]
        and capital_preserving_slot_diag["reserved_cost_milli"]
        >= capital_preserving_slot_diag["baseline_reserved_cost_milli"]
        and capital_preserving_slot_selected["selected_rows"][0]["ticker"] == "Q1",
    )

    capital_preserving_reject_rows = [
        _resource_candidate_fixed(
            "R1", 1000.0, 110, 0.10,
            "resource-aware-continuous-capital-preserving",
        ),
        _resource_candidate_fixed(
            "Q1", 1000.0, 80, 0.95,
            "resource-aware-continuous-capital-preserving",
        ),
    ]
    capital_preserving_reject_order, capital_preserving_reject_diag = (
        reorder_candidates_for_resource_aware_quality(
            capital_preserving_reject_rows,
            available_cash=180_000.0,
            sizing_equity=2_000_000.0,
            pre_market_occupied=9,
            max_positions=10,
            params=resource_params,
        )
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capital_preserving_continuous_rejects_higher_score_basket_when_reserved_capital_would_fall",
        True,
        capital_preserving_reject_diag["mode"] == "dl-selection"
        and not capital_preserving_reject_diag["changed"]
        and capital_preserving_reject_diag["resource_preservation_required"]
        and capital_preserving_reject_diag["selected_count_preserved"]
        and capital_preserving_reject_diag["reserved_capital_preserved"]
        and capital_preserving_reject_diag["reserved_cost_milli"]
        == capital_preserving_reject_diag["baseline_reserved_cost_milli"]
        and [row["ticker"] for row in capital_preserving_reject_order]
        == ["R1", "Q1"],
    )

    max_dl_seed = (
        ("T0", 300.0, 259, 0.791),
        ("T1", 100.0, 121, 0.522),
        ("T2", 1000.0, 303, 0.557),
        ("T3", 300.0, 296, 0.105),
        ("T4", 300.0, 155, 0.893),
        ("T5", 300.0, 87, 0.272),
        ("T6", 1000.0, 164, 0.203),
    )
    max_dl_rows = [
        _resource_candidate_fixed(
            ticker,
            price,
            qty,
            score,
            "resource-aware-continuous-max-dl",
        )
        for ticker, price, qty, score in max_dl_seed
    ]
    max_dl_baseline = _simulate_reserved_candidate_order(
        max_dl_rows,
        available_cash=500_000.0,
        sizing_equity=2_000_000.0,
        free_slots=3,
        params=resource_params,
    )
    max_dl_order, max_dl_diag = reorder_candidates_for_resource_aware_quality(
        max_dl_rows,
        available_cash=500_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=7,
        max_positions=10,
        params=resource_params,
    )
    max_dl_action_rows = select_resource_aware_action_candidates(
        max_dl_order,
        max_dl_diag,
    )
    max_dl_selected = _simulate_reserved_candidate_order(
        max_dl_action_rows,
        available_cash=500_000.0,
        sizing_equity=2_000_000.0,
        free_slots=3,
        params=resource_params,
    )
    max_dl_base_rank = {id(row): idx for idx, row in enumerate(max_dl_rows)}
    max_dl_oracle = None
    for combo in itertools.combinations(max_dl_rows, 3):
        execution_order = sorted(combo, key=lambda row: max_dl_base_rank[id(row)])
        trial = _simulate_reserved_candidate_order(
            execution_order,
            available_cash=500_000.0,
            sizing_equity=2_000_000.0,
            free_slots=3,
            params=resource_params,
        )
        if (
            trial["selected_count"] != max_dl_baseline["selected_count"]
            or trial["reserved_cost_milli"] < max_dl_baseline["reserved_cost_milli"]
        ):
            continue
        score_sum = sum(float(row["breakout_quality_score"]) for row in combo)
        oracle_key = (
            score_sum,
            tuple(-max_dl_base_rank[id(row)] for row in execution_order),
        )
        if max_dl_oracle is None or oracle_key > max_dl_oracle[0]:
            max_dl_oracle = (
                oracle_key,
                [row["ticker"] for row in execution_order],
            )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_constrained_basket_uses_fixed_min_roos_k_and_exact_reserve_floor_with_independent_bruteforce_matched_one_step_repair",
        True,
        max_dl_oracle is not None
        and max_dl_diag["mode"] == "dl-selection"
        and max_dl_diag["max_dl_eligible"]
        and max_dl_diag["max_dl_repair_steps"] == 1
        and not max_dl_diag["max_dl_fallback_to_baseline"]
        and max_dl_diag["pre_market_order_limit"]
        == max_dl_baseline["selected_count"]
        and len(max_dl_order) == len(max_dl_rows)
        and len(max_dl_action_rows) == max_dl_baseline["selected_count"]
        and max_dl_selected["selected_count"] == max_dl_baseline["selected_count"]
        and max_dl_selected["reserved_cost_milli"]
        >= max_dl_baseline["reserved_cost_milli"]
        and [row["ticker"] for row in max_dl_action_rows] == max_dl_oracle[1]
        and [row["ticker"] for row in max_dl_action_rows] == ["T0", "T2", "T6"],
    )
    max_dl_mechanism = build_max_dl_repair_mechanism_diagnostic(
        max_dl_rows,
        available_cash=500_000.0,
        sizing_equity=2_000_000.0,
        params=resource_params,
        resource_selection_diag=max_dl_diag,
    )
    repair_trace = list((max_dl_mechanism or {}).get("repair_steps_trace") or [])
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_one_step_repair_certificate_reuses_exhaustive_production_single_swap_search_without_global_combinatorial_oracle",
        True,
        bool(
            max_dl_mechanism is not None
            and max_dl_mechanism.get("status") == "AVAILABLE"
            and max_dl_mechanism.get("actual_repair_steps") == 1
            and max_dl_mechanism.get("actual_repair_replacement_distance") == 1
            and max_dl_mechanism.get("classification") == "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT"
            and len(repair_trace) == 1
            and int(repair_trace[0].get("evaluated_swap_count", 0)) > 0
            and int(repair_trace[0].get("feasible_swap_count", 0)) > 0
            and bool(repair_trace[0].get("after_feasible", False))
            and "global_best_score_sum" not in max_dl_mechanism
            and "exact_oracle_evaluated_states" not in max_dl_mechanism
            and math.isclose(
                float(max_dl_mechanism.get("repair_seed_score_sum")),
                float(max_dl_oracle[0][0]),
                abs_tol=1e-12,
            )
        ),
    )

    max_dl_direct_rows = [
        _resource_candidate_fixed(
            "D0", 1000.0, 120, 0.20,
            "resource-aware-continuous-max-dl",
        ),
        _resource_candidate_fixed(
            "D1", 1000.0, 120, 0.95,
            "resource-aware-continuous-max-dl",
        ),
        _resource_candidate_fixed(
            "D2", 1000.0, 120, 0.90,
            "resource-aware-continuous-max-dl",
        ),
    ]
    max_dl_direct_order, max_dl_direct_diag = reorder_candidates_for_resource_aware_quality(
        max_dl_direct_rows,
        available_cash=250_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    max_dl_direct_action = select_resource_aware_action_candidates(
        max_dl_direct_order,
        max_dl_direct_diag,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_constrained_basket_keeps_full_orderable_universe_but_only_action_prefix_can_create_orders",
        True,
        max_dl_direct_diag["direct_score_order_feasible"]
        and max_dl_direct_diag["pre_market_order_limit"] == 2
        and len(max_dl_direct_order) == 3
        and len(max_dl_direct_action) == 2
        and {row["ticker"] for row in max_dl_direct_action} == {"D1", "D2"},
    )


    ascent_gap_seed = (
        ("X0", 300.0, 119, 0.027),
        ("X1", 200.0, 304, 0.206),
        ("X2", 100.0, 287, 0.827),
        ("X3", 1000.0, 93, 0.817),
        ("X4", 500.0, 233, 0.056),
        ("X5", 500.0, 283, 0.379),
        ("X6", 300.0, 114, 0.892),
    )
    ascent_gap_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
        )
        for ticker, price, qty, score in ascent_gap_seed
    ]
    ascent_gap_order, ascent_gap_diag = reorder_candidates_for_resource_aware_quality(
        ascent_gap_rows,
        available_cash=300_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=7,
        max_positions=10,
        params=resource_params,
    )
    ascent_gap_action = select_resource_aware_action_candidates(
        ascent_gap_order, ascent_gap_diag
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_feasible_ascent_improves_c17_local_repair_without_changing_k_or_resource_floor",
        True,
        [row["ticker"] for row in ascent_gap_action] == ["X1", "X3", "X5"]
        and ascent_gap_diag["max_dl_feasible_ascent_steps"] > 0
        and ascent_gap_diag["max_dl_feasible_ascent_evaluations"] > 0
        and ascent_gap_diag["max_dl_feasible_ascent_local_optimum"]
        and not ascent_gap_diag["max_dl_fallback_to_baseline"]
        and ascent_gap_diag["selector_elapsed_ns"] > 0
        and ascent_gap_diag["selected_count"] == ascent_gap_diag["baseline_selected_count"]
        and ascent_gap_diag["reserved_cost_milli"] >= ascent_gap_diag["baseline_reserved_cost_milli"],
    )
    trace_baskets = dict(ascent_gap_diag.get("_selector_trace_baskets") or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_selector_exposes_transient_raw_repair_and_ascent_membership_without_changing_final_action",
        True,
        bool(
            [row["ticker"] for row in trace_baskets.get("raw_top_n", [])]
            and len(trace_baskets.get("raw_top_n", [])) == ascent_gap_diag["pre_market_order_limit"]
            and len(trace_baskets.get("minimum_repair_seed", [])) == ascent_gap_diag["pre_market_order_limit"]
            and [row["ticker"] for row in trace_baskets.get("feasible_ascent_final", [])]
            == [row["ticker"] for row in ascent_gap_action]
        ),
    )

    fallback_ascent_seed = (
        ("F0", 200.0, 153, 0.899),
        ("F1", 500.0, 249, 0.312),
        ("F2", 1000.0, 233, 0.346),
        ("F3", 1000.0, 162, 0.685),
        ("F4", 100.0, 334, 0.012),
        ("F5", 100.0, 326, 0.929),
        ("F6", 1000.0, 51, 0.706),
        ("F7", 300.0, 283, 0.585),
    )
    fallback_c17_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score, "resource-aware-continuous-max-dl"
        )
        for ticker, price, qty, score in fallback_ascent_seed
    ]
    fallback_c18_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
        )
        for ticker, price, qty, score in fallback_ascent_seed
    ]
    fallback_c17_order, fallback_c17_diag = reorder_candidates_for_resource_aware_quality(
        fallback_c17_rows,
        available_cash=400_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    fallback_c18_order, fallback_c18_diag = reorder_candidates_for_resource_aware_quality(
        fallback_c18_rows,
        available_cash=400_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    fallback_c17_action = select_resource_aware_action_candidates(
        fallback_c17_order, fallback_c17_diag
    )
    fallback_c18_action = select_resource_aware_action_candidates(
        fallback_c18_order, fallback_c18_diag
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_feasible_ascent_continues_dl_optimization_after_c17_seed_fallback",
        True,
        fallback_c17_diag["max_dl_fallback_to_baseline"]
        and fallback_c18_diag["max_dl_seed_fallback"]
        and not fallback_c18_diag["max_dl_fallback_to_baseline"]
        and fallback_c18_diag["max_dl_feasible_ascent_steps"] > 0
        and fallback_c18_diag["selected_score_sum"] > fallback_c17_diag["selected_score_sum"]
        and [row["ticker"] for row in fallback_c17_action] == ["F1", "F2"]
        and [row["ticker"] for row in fallback_c18_action] == ["F2", "F3"]
        and fallback_c18_diag["selected_count"] == fallback_c18_diag["baseline_selected_count"]
        and fallback_c18_diag["reserved_cost_milli"] >= fallback_c18_diag["baseline_reserved_cost_milli"],
    )


    ascent_gap_seed = (
        ("X0", 300.0, 119, 0.027),
        ("X1", 200.0, 304, 0.206),
        ("X2", 100.0, 287, 0.827),
        ("X3", 1000.0, 93, 0.817),
        ("X4", 500.0, 233, 0.056),
        ("X5", 500.0, 283, 0.379),
        ("X6", 300.0, 114, 0.892),
    )
    ascent_gap_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
        )
        for ticker, price, qty, score in ascent_gap_seed
    ]
    ascent_gap_order, ascent_gap_diag = reorder_candidates_for_resource_aware_quality(
        ascent_gap_rows,
        available_cash=300_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=7,
        max_positions=10,
        params=resource_params,
    )
    ascent_gap_action = select_resource_aware_action_candidates(
        ascent_gap_order, ascent_gap_diag
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_feasible_ascent_improves_c17_local_repair_without_changing_k_or_resource_floor",
        True,
        [row["ticker"] for row in max_dl_action_rows] == ["T0", "T2", "T6"]
        and [row["ticker"] for row in ascent_gap_action] == ["X1", "X3", "X5"]
        and ascent_gap_diag["max_dl_feasible_ascent_steps"] > 0
        and ascent_gap_diag["max_dl_feasible_ascent_evaluations"] > 0
        and ascent_gap_diag["max_dl_feasible_ascent_local_optimum"]
        and not ascent_gap_diag["max_dl_fallback_to_baseline"]
        and ascent_gap_diag["selector_elapsed_ns"] > 0
        and ascent_gap_diag["selected_count"] == ascent_gap_diag["baseline_selected_count"]
        and ascent_gap_diag["reserved_cost_milli"] >= ascent_gap_diag["baseline_reserved_cost_milli"],
    )

    fallback_ascent_seed = (
        ("F0", 200.0, 153, 0.899),
        ("F1", 500.0, 249, 0.312),
        ("F2", 1000.0, 233, 0.346),
        ("F3", 1000.0, 162, 0.685),
        ("F4", 100.0, 334, 0.012),
        ("F5", 100.0, 326, 0.929),
        ("F6", 1000.0, 51, 0.706),
        ("F7", 300.0, 283, 0.585),
    )
    fallback_c17_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score, "resource-aware-continuous-max-dl"
        )
        for ticker, price, qty, score in fallback_ascent_seed
    ]
    fallback_c18_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
        )
        for ticker, price, qty, score in fallback_ascent_seed
    ]
    fallback_c17_order, fallback_c17_diag = reorder_candidates_for_resource_aware_quality(
        fallback_c17_rows,
        available_cash=400_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    fallback_c18_order, fallback_c18_diag = reorder_candidates_for_resource_aware_quality(
        fallback_c18_rows,
        available_cash=400_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=6,
        max_positions=10,
        params=resource_params,
    )
    fallback_c17_action = select_resource_aware_action_candidates(
        fallback_c17_order, fallback_c17_diag
    )
    fallback_c18_action = select_resource_aware_action_candidates(
        fallback_c18_order, fallback_c18_diag
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "max_dl_feasible_ascent_continues_dl_optimization_after_c17_seed_fallback",
        True,
        fallback_c17_diag["max_dl_fallback_to_baseline"]
        and fallback_c18_diag["max_dl_seed_fallback"]
        and not fallback_c18_diag["max_dl_fallback_to_baseline"]
        and fallback_c18_diag["max_dl_feasible_ascent_steps"] > 0
        and fallback_c18_diag["selected_score_sum"] > fallback_c17_diag["selected_score_sum"]
        and [row["ticker"] for row in fallback_c17_action] == ["F1", "F2"]
        and [row["ticker"] for row in fallback_c18_action] == ["F2", "F3"]
        and fallback_c18_diag["selected_count"] == fallback_c18_diag["baseline_selected_count"]
        and fallback_c18_diag["reserved_cost_milli"] >= fallback_c18_diag["baseline_reserved_cost_milli"],
    )

    mocked_pair_payload = {
        "metadata": {
            "comparison_mode": "score-ranking",
            "comparison_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "params_path": "models/min_roos.json",
            "param_source_kind": "rolling_active_param_ensemble",
            "param_selector": "base_finalist_best",
            "runtime_member_count_min": 1,
            "runtime_member_count_max": 1,
            "runtime_min_agree": 1,
            "comparison_design": "synthetic_score_ranking",
            "lookahead_safe_active_param_schedule": True,
            "dataset": "full",
            "score_source": "continuous_ranker_oos",
            "score_ranking_policy": "resource-aware-continuous-max-dl",
            "optional_entry_filter_policy": "all-off",
            "benchmark_ticker": "0050",
            "filter_id": "breakout_quality_v1",
            "model_architecture": "inception_time_v1",
            "experiment_profile": "strategy_aligned_no_time_all_event_mse",
            "score_ranking_order": ["continuous_score_top_k"],
            "ranking_scope": "all_candidates_after_single_member_qualification",
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
        "score_ranking_minus_no_filter": {
            "total_return_pct": 2.0,
            "max_drawdown_pct": 0.0,
            "return_over_max_drawdown": 0.4,
            "annual_return_pct": 2.0,
            "expected_value_r": 0.03,
            "payoff_ratio": 0.10,
            "avg_exposure_pct": 0.0,
            "trade_count": 0,
        },
        "yearly": [
            {
                "year": 2021,
                "no_filter_return_pct": 10.0,
                "quality_filter_return_pct": 11.0,
                "score_ranking_return_pct": 12.0,
                "delta_pct": 2.0,
                "is_full_year": True,
            }
        ],
    }
    from filters.breakout_quality.strategy_compare_engine import (
        materialize_strategy_pair_readable_report,
        render_strategy_pair_simple_report,
    )
    rendered_pair_summary = render_strategy_pair_simple_report(
        mocked_pair_payload,
        color=False,
    )
    with tempfile.TemporaryDirectory() as report_tmp:
        readable_report_path = materialize_strategy_pair_readable_report(
            mocked_pair_payload,
            output_dir=Path(report_tmp),
        )
        readable_report_text = readable_report_path.read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_pair_outputs_always_materialize_markdown_and_console_simple_report",
        True,
        "Breakout Quality Score 排序策略經濟效果對照" in rendered_pair_summary
        and "主要結果" in rendered_pair_summary
        and readable_report_path.name == "strategy_comparison.md"
        and "# Breakout Quality Score 排序策略經濟效果對照" in readable_report_text,
    )

    mocked_baseline_payload = {
        "metadata": {
            **dict(mocked_pair_payload["metadata"]),
            "comparison_design": "standalone_dl_off_active_param_replay",
            "score_source": "dl_off_baseline_only",
        },
        "no_filter": dict(mocked_pair_payload["no_filter"]),
        "yearly": [
            {
                "year": row["year"],
                "no_filter_return_pct": row["no_filter_return_pct"],
                "is_full_year": row.get("is_full_year", False),
            }
            for row in mocked_pair_payload["yearly"]
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
        "run_standalone_baseline",
        return_value=dict(mocked_baseline_payload),
    ) as mocked_baseline_run, patch.object(
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
        and mocked_baseline_run.call_count
        == len(strategy_comparison_module._standalone_baseline_arms(settings))
        and all(
            call.kwargs.get("comparison_start_date") == "2021-01-01"
            and call.kwargs.get("comparison_end_date") == "2021-12-31"
            and call.kwargs.get("score_source")
            == settings.dl_sources[pair_contract[3].dl_id].score_source
            for pair_contract, call in zip(execution_pairs, mocked_run.call_args_list)
        )
        and set(replay_payload["scenarios"])
        == {arm.arm_id for arm in settings.enabled_arms}
        and replay_payload["status"] == "COMPLETED",
    )

    seen_baseline_groups = set()
    baseline_reuse_contract_ok = True
    for pair_contract, call in zip(execution_pairs, mocked_run.call_args_list):
        param_source, rule_policy, _off_arm, _on_arm = pair_contract
        group_key = (param_source, rule_policy)
        baseline_reuse_dir = call.kwargs.get("baseline_reuse_dir")
        if (
            settings.preparation.reuse_shared_baseline
            and group_key in seen_baseline_groups
        ):
            baseline_reuse_contract_ok = (
                baseline_reuse_contract_ok
                and baseline_reuse_dir is not None
            )
        else:
            baseline_reuse_contract_ok = (
                baseline_reuse_contract_ok
                and baseline_reuse_dir is None
            )
        seen_baseline_groups.add(group_key)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "same_run_multiple_dl_arms_reuse_one_shared_baseline_after_first_pair",
        True,
        baseline_reuse_contract_ok,
    )

    cache_off = settings.arms.get("C3")
    cache_on = settings.arms.get("C17")
    if cache_off is not None and cache_on is not None and cache_on.dl_id:
        with tempfile.TemporaryDirectory() as required_tmp:
            required_names = {
                path.name
                for path in strategy_comparison_module._pair_cache_required_files(
                    Path(required_tmp),
                    on_arm=cache_on,
                )
            }
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "completed_pair_cache_requires_human_readable_strategy_report",
            True,
            "strategy_comparison.md" in required_names
            and "strategy_comparison.json" in required_names,
        )
        cache_period = {"start": "2021-01-01", "end": "2025-12-22"}
        cache_artifacts = {
            "param:min_roos": {"path": "models/min.json", "sha256": "param-sha"},
            "dl:CONT12A:model": {"path": "models/cont12a.pt", "sha256": "model-sha"},
            "dl:CONT12A:manifest": {"path": "models/cont12a.json", "sha256": "manifest-sha"},
            "dl:CONT12A:forward_scores": {"path": "models/cont12a.csv", "sha256": "scores-sha"},
        }
        stored_settings = settings.as_dict()
        pair_group = strategy_comparison_module._pair_group_id(
            param_source=cache_on.param_source,
            rule_policy=cache_on.rule_policy,
            dl_id=cache_on.dl_id,
            dl_runtime_mode=str(cache_on.dl_runtime_mode or ""),
        )
        cached_pair_payload = {
            "metadata": {
                "schema_version": strategy_comparison_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
                "comparison_period": cache_period,
            },
            "no_filter": {},
            "score_ranking": {},
            "yearly": [],
        }
        with tempfile.TemporaryDirectory() as cache_tmp:
            cache_root = Path(cache_tmp)
            cache_read_root = (
                settings.reuse_output_roots[0]
                if settings.reuse_output_roots
                else settings.output_root
            )
            cached_run = (
                cache_root
                / cache_read_root
                / "runs"
                / "20260808_000000_C3-C17_cache"
            )
            cached_pair_dir = cached_run / "pairs" / pair_group
            cached_pair_dir.mkdir(parents=True, exist_ok=True)
            (cached_pair_dir / "strategy_comparison.json").write_text(
                json.dumps(cached_pair_payload),
                encoding="utf-8",
            )
            for filename in (
                "strategy_comparison.md",
                "yearly_returns_comparison.csv",
                "no_filter_equity.csv",
                "score_ranking_equity.csv",
                "no_filter_trades.csv",
                "score_ranking_trades.csv",
                "no_filter_daily_capacity.csv",
                "score_ranking_daily_capacity.csv",
                "no_filter_orderable_candidates.csv",
                "score_ranking_orderable_candidates.csv",
                "no_filter_selected_buys.csv",
                "score_ranking_selected_buys.csv",
                "score_ranking_execution.csv",
                "score_ranking_selector_trace.csv",
                "score_ranking_repair_search_certificate.csv",
            ):
                (cached_pair_dir / filename).write_text("x\n", encoding="utf-8")
            (cached_run / "strategy_comparison.json").write_text(
                json.dumps({
                    "status": "COMPLETED",
                    "settings": stored_settings,
                    "artifact_identities": cache_artifacts,
                    "comparison_period": cache_period,
                    "pairs": {pair_group: cached_pair_payload},
                }),
                encoding="utf-8",
            )
            cache_status = {
                "artifact_identities": cache_artifacts,
                "comparison_period": cache_period,
            }
            cache_hit = strategy_comparison_module._find_reusable_pair(
                root=cache_root,
                settings=settings,
                status=cache_status,
                off_arm=cache_off,
                on_arm=cache_on,
            )
            changed_artifacts = dict(cache_artifacts)
            changed_artifacts["dl:CONT12A:forward_scores"] = {
                "path": "models/cont12a.csv",
                "sha256": "scores-sha-changed",
            }
            cache_miss_after_score_change = (
                strategy_comparison_module._find_reusable_pair(
                    root=cache_root,
                    settings=settings,
                    status={
                        "artifact_identities": changed_artifacts,
                        "comparison_period": cache_period,
                    },
                    off_arm=cache_off,
                    on_arm=cache_on,
                )
                is None
            )
        settings_without_report_identity = settings.as_dict()
        settings_without_report_identity["contrasts"] = {
            "synthetic-only": {
                "enabled": True,
                "left": "C17",
                "right": "C3",
                "description": "report-only change",
            }
        }
        current_fp = strategy_comparison_module._pair_cache_fingerprint_from_payload(
            settings_payload=settings.as_dict(),
            artifact_identities=cache_artifacts,
            comparison_period=cache_period,
            off_arm_payload=cache_off.as_dict(),
            on_arm_payload=cache_on.as_dict(),
            engine_schema_version=strategy_comparison_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
        )
        report_changed_fp = strategy_comparison_module._pair_cache_fingerprint_from_payload(
            settings_payload=settings_without_report_identity,
            artifact_identities=cache_artifacts,
            comparison_period=cache_period,
            off_arm_payload=cache_off.as_dict(),
            on_arm_payload=cache_on.as_dict(),
            engine_schema_version=strategy_comparison_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "completed_pair_cache_uses_replay_identity_not_whole_config_and_invalidates_on_score_sha",
            True,
            cache_hit is not None
            and current_fp == report_changed_fp
            and cache_miss_after_score_change,
        )

        historical_ids = {"C3", "C17", "C18", "C19", "C20", "C21", "C22"}
        if historical_ids.issubset(set(settings.arms)):
            historical_settings = replace(
                settings,
                arms={
                    arm_id: replace(arm, enabled=arm_id in historical_ids)
                    for arm_id, arm in settings.arms.items()
                },
                contrasts={
                    contrast_id: replace(contrast, enabled=False)
                    for contrast_id, contrast in settings.contrasts.items()
                },
            )
            with tempfile.TemporaryDirectory() as reuse_tmp:
                reuse_root = Path(reuse_tmp)
                reusable_pair_dir = reuse_root / "historical_pair"
                reusable_pair_dir.mkdir(parents=True, exist_ok=True)
                (reusable_pair_dir / "strategy_comparison.json").write_text(
                    json.dumps(mocked_pair_payload),
                    encoding="utf-8",
                )
                reuse_status = dict(ready_status)
                reuse_status["resolved_parameter_paths"] = {
                    "min_roos": "models/min_roos.json"
                }
                reuse_status["replay_cache"] = {
                    "pairs": {
                        "C17": {
                            "source_pair_dir": reusable_pair_dir,
                            "fingerprint": "c17-cache",
                        },
                        "C18": {
                            "source_pair_dir": reusable_pair_dir,
                            "fingerprint": "c18-cache",
                        },
                        "C19": {
                            "source_pair_dir": reusable_pair_dir,
                            "fingerprint": "c19-cache",
                        },
                        "C20": {
                            "source_pair_dir": reusable_pair_dir,
                            "fingerprint": "c20-cache",
                        },
                        "C21": None,
                        "C22": None,
                    },
                    "baseline_groups": {
                        "min_roos::all_off": {
                            "off_arm_id": "C3",
                            "source_pair_dir": reusable_pair_dir,
                        }
                    },
                }
                reuse_console = io.StringIO()
                with patch.object(
                    strategy_comparison_module,
                    "get_strategy_comparison_settings",
                    return_value=historical_settings,
                ), patch.object(
                    strategy_comparison_module,
                    "run_comparison",
                    return_value=dict(mocked_pair_payload),
                ) as cache_run, patch.object(
                    strategy_comparison_module,
                    "_load_direct_selection_r",
                    return_value=0.25,
                ), redirect_stdout(reuse_console):
                    cached_execution_payload = (
                        strategy_comparison_module.run_strategy_comparison(
                            project_root=reuse_root,
                            quiet=False,
                            status=reuse_status,
                            auto_prepare=False,
                        )
                    )
                cache_actions = {
                    key: value["action"]
                    for key, value in cached_execution_payload[
                        "pair_execution"
                    ].items()
                }
                only_new_model_pairs_run = (
                    cache_run.call_count == 2
                    and cache_actions.get("C17") == "REUSE"
                    and cache_actions.get("C18") == "REUSE"
                    and cache_actions.get("C19") == "REUSE"
                    and cache_actions.get("C20") == "REUSE"
                    and cache_actions.get("C21") == "RUN"
                    and cache_actions.get("C22") == "RUN"
                    and all(
                        call.kwargs.get("baseline_reuse_dir") is not None
                        for call in cache_run.call_args_list
                    )
                    and reuse_console.getvalue().count(
                        "Breakout Quality Score 排序策略經濟效果對照"
                    ) >= 2
                    and all(
                        (
                            reuse_root
                            / str(cached_execution_payload["pair_execution"][arm_id]["current_pair_dir"])
                            / "strategy_comparison.md"
                        ).is_file()
                        for arm_id in ("C17", "C18", "C19", "C20")
                    )
                )
            add_check(
                results, "synthetic_breakout_quality", case_id,
                "multi_model_matrix_reuses_c17_c20_and_runs_only_c21_c22_with_shared_baseline",
                True,
                only_new_model_pairs_run,
            )

    from filters.breakout_quality.strategy_compare_engine import (
        _load_reusable_no_filter_baseline,
        COMPARISON_MODE_SCORE_RANKING as _CACHE_SCORE_RANKING_MODE,
    )
    with tempfile.TemporaryDirectory() as baseline_tmp:
        baseline_dir = Path(baseline_tmp)
        baseline_metadata = {
            "schema_version": strategy_comparison_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
            "dataset": "full",
            "params_file_sha256": "param-sha",
            "requested_param_policy": "base-finalist-best",
            "optional_entry_filter_policy": "all-off",
            "shared_param_overrides": {"use_bb": False},
            "max_positions": 10,
            "enable_rotation": False,
            "comparison_period": {
                "start": "2021-01-01",
                "end": "2021-12-31",
            },
        }
        (baseline_dir / "strategy_comparison.json").write_text(
            json.dumps({
                "metadata": baseline_metadata,
                "no_filter": {
                    "total_return_pct": 10.0,
                    "benchmark_return_pct": 5.0,
                    "benchmark_max_drawdown_pct": 4.0,
                    "benchmark_annual_return_pct": 5.0,
                },
            }),
            encoding="utf-8",
        )
        pd.DataFrame({
            "Date": ["2021-01-04", "2021-01-05"],
            "Equity": [1_000_000.0, 1_001_000.0],
        }).to_csv(
            baseline_dir / "no_filter_equity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({
            "Date": ["2021-01-05"],
            "Ticker": ["2330"],
        }).to_csv(
            baseline_dir / "no_filter_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({
            "Date": ["2021-01-04"],
            "Positions": [1],
        }).to_csv(
            baseline_dir / "no_filter_daily_capacity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({
            "date": ["2021-01-04"],
            "ticker": ["2330"],
        }).to_csv(
            baseline_dir / "no_filter_orderable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame({
            "year": [2021],
            "no_filter_return_pct": [10.0],
            "score_ranking_return_pct": [12.0],
            "is_full_year": [True],
            "start_date": ["2021-01-01"],
            "end_date": ["2021-12-31"],
        }).to_csv(
            baseline_dir / "yearly_returns_comparison.csv",
            index=False,
            encoding="utf-8-sig",
        )
        baseline_payload, baseline_summary, baseline_orderable = (
            _load_reusable_no_filter_baseline(
                baseline_dir,
                comparison_mode=_CACHE_SCORE_RANKING_MODE,
                expected_dataset="full",
                expected_params_sha256="param-sha",
                expected_param_policy="base-finalist-best",
                expected_optional_entry_filter_policy="all-off",
                expected_shared_param_overrides={"use_bb": False},
                expected_max_positions=10,
                expected_enable_rotation=False,
                expected_start_date="2021-01-01",
                expected_end_date="2021-12-31",
            )
        )
        try:
            _load_reusable_no_filter_baseline(
                baseline_dir,
                comparison_mode=_CACHE_SCORE_RANKING_MODE,
                expected_dataset="full",
                expected_params_sha256="changed-param-sha",
                expected_param_policy="base-finalist-best",
                expected_optional_entry_filter_policy="all-off",
                expected_shared_param_overrides={"use_bb": False},
                expected_max_positions=10,
                expected_enable_rotation=False,
                expected_start_date="2021-01-01",
                expected_end_date="2021-12-31",
            )
        except ValueError:
            mismatched_baseline_rejected = True
        else:
            mismatched_baseline_rejected = False
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "shared_baseline_reuse_rehydrates_canonical_artifacts_and_rejects_contract_mismatch",
        True,
        baseline_summary.get("total_return_pct") == 10.0
        and len(baseline_payload["equity_curve"]) == 2
        and len(baseline_orderable) == 1
        and mismatched_baseline_rejected,
    )

    # Exercise shared-baseline aggregation with an isolated synthetic enabled
    # matrix. Historical arms may legitimately be disabled by the user's current
    # config, so this contract must not depend on the active comparison matrix.
    registered_groups: dict[tuple[str, str], dict[str, object]] = {}
    registered_group_order: list[tuple[str, str]] = []
    for arm in settings.arms.values():
        key = (arm.param_source, arm.rule_policy)
        if key not in registered_groups:
            registered_groups[key] = {"off": None, "on": []}
            registered_group_order.append(key)
        group = registered_groups[key]
        if arm.dl_enabled:
            group["on"].append(arm)
        elif group["off"] is None:
            group["off"] = arm

    shared_group = next(
        (
            registered_groups[key]
            for key in registered_group_order
            if registered_groups[key]["off"] is not None
            and len(registered_groups[key]["on"]) >= 2
        ),
        None,
    )
    if shared_group is None:
        shared_baseline_fixture_available = False
        repeated_baseline_mismatch_rejected = False
        volatile_timing_ignored = False
    else:
        shared_baseline_fixture_available = True
        selected_off = shared_group["off"]
        selected_on = list(shared_group["on"][:2])
        enabled_ids = {selected_off.arm_id, *(arm.arm_id for arm in selected_on)}
        synthetic_arms = {
            arm_id: replace(arm, enabled=arm_id in enabled_ids)
            for arm_id, arm in settings.arms.items()
        }
        synthetic_settings = replace(settings, arms=synthetic_arms)
        synthetic_execution_pairs = strategy_comparison_module._execution_pairs(
            synthetic_settings
        )
        pair_a = synthetic_execution_pairs[0]
        pair_b = synthetic_execution_pairs[1]
        mismatched_pairs = {
            "shared_a": {
                "arm_contract": pair_a,
                "payload": dict(mocked_pair_payload),
            },
            "shared_b": {
                "arm_contract": pair_b,
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
                {"shared_a": 0.1, "shared_b": 0.2},
                settings=synthetic_settings,
            )
        except ValueError as exc:
            repeated_baseline_mismatch_rejected = "共用基準不一致" in str(exc)
        else:
            repeated_baseline_mismatch_rejected = False

        timing_only_pairs = {
            "shared_a": {
                "arm_contract": pair_a,
                "payload": {
                    **dict(mocked_pair_payload),
                    "no_filter": {
                        **dict(mocked_pair_payload["no_filter"]),
                        "resource_aware_selector_timing_calls": 1,
                        "resource_aware_selector_timing_total_ms": 0.050,
                        "resource_aware_selector_timing_median_ms": 0.050,
                        "resource_aware_selector_timing_p95_ms": 0.050,
                        "resource_aware_selector_timing_max_ms": 0.050,
                    },
                },
            },
            "shared_b": {
                "arm_contract": pair_b,
                "payload": {
                    **dict(mocked_pair_payload),
                    "no_filter": {
                        **dict(mocked_pair_payload["no_filter"]),
                        "resource_aware_selector_timing_calls": 1,
                        "resource_aware_selector_timing_total_ms": 0.091,
                        "resource_aware_selector_timing_median_ms": 0.091,
                        "resource_aware_selector_timing_p95_ms": 0.091,
                        "resource_aware_selector_timing_max_ms": 0.091,
                    },
                },
            },
        }
        try:
            timing_scenarios = strategy_comparison_module._scenario_payloads(
                timing_only_pairs,
                {"shared_a": 0.1, "shared_b": 0.2},
                settings=synthetic_settings,
            )
        except ValueError:
            volatile_timing_ignored = False
        else:
            volatile_timing_ignored = (
                timing_scenarios.get(selected_off.arm_id, {}).get("total_return_pct")
                == 10.0
            )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multiple_dl_sources_require_identical_replayed_shared_baseline",
        True,
        shared_baseline_fixture_available and repeated_baseline_mismatch_rejected,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "shared_baseline_consistency_ignores_only_volatile_selector_cpu_timing",
        True,
        shared_baseline_fixture_available and volatile_timing_ignored,
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
        and "重新規劃後沒有可執行且依賴已就緒的動作" in preparation_source,
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
        builder_type="binary_dl_min_roos_rolling",
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

    dependency_reuse = StrategyPreparationAction(
        action_id="dataset:truth", artifact_key="dataset:truth",
        action="REUSE", builder_type=None, description="reuse truth", path="outputs/dataset.json",
        producer_work_type="existing_artifact",
    )
    dependency_build = StrategyPreparationAction(
        action_id="score:forward", artifact_key="score:forward",
        action="BUILD", builder_type="score_builder", description="build score", path="outputs/score.csv",
        dependencies=("dataset:truth",), producer_work_type="strategy_compare_deterministic_rebuild",
    )
    dependency_plan = StrategyPreparationPlan.from_actions(
        (dependency_build, dependency_reuse)
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_declares_dependency_and_producer_contract",
        True,
        dependency_plan.overall_status == "PREPARABLE"
        and dependency_plan.next_runnable_action() is dependency_build
        and dependency_build.as_dict()["dependencies"] == ["dataset:truth"]
        and dependency_build.as_dict()["producer_work_type"]
        == "strategy_compare_deterministic_rebuild",
    )

    dependency_cycle_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="a", artifact_key="a", action="BUILD", builder_type="x",
                description="a", path="a", dependencies=("b",),
            ),
            StrategyPreparationAction(
                action_id="b", artifact_key="b", action="BUILD", builder_type="x",
                description="b", path="b", dependencies=("a",),
            ),
        ))
    except ValueError:
        dependency_cycle_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_dependency_cycles",
        True,
        dependency_cycle_rejected,
    )

    unknown_dependency_rejected = False
    try:
        StrategyPreparationPlan.from_actions((
            StrategyPreparationAction(
                action_id="score", artifact_key="score", action="BUILD", builder_type="x",
                description="score", path="score", dependencies=("missing:truth",),
            ),
        ))
    except ValueError:
        unknown_dependency_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "preparation_plan_rejects_unknown_dependencies",
        True,
        unknown_dependency_rejected,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "parameter_preflight_identity_tracks_current_min_roos_contract_not_removed_full_baseline",
        True,
        all(token in preparation_source for token in (
            "TRAINING_CONFIG_MISMATCH", "BINARY_PIT_IDENTITY_MISSING",
            "MIN_ROOS_SEARCH_FIELDS_MISMATCH", "trials_per_fold",
            "max_position_cap_pct",
        ))
        and "BASELINE_PARAMS_IDENTITY_MISMATCH" not in preparation_source,
    )

    enabled_dl_arm_ids = [
        arm.arm_id for arm in settings.enabled_arms if arm.dl_enabled
    ]
    switch_target_arm_id = enabled_dl_arm_ids[0] if len(enabled_dl_arm_ids) >= 2 else None
    profile_id = settings.profile_id
    original_profile = dict(strategy_config.STRATEGY_COMPARE_PROFILES[profile_id])
    reduced_settings = settings
    if switch_target_arm_id is not None:
        try:
            reduced_arm_ids = tuple(
                arm_id
                for arm_id in original_profile["arm_ids"]
                if arm_id != switch_target_arm_id
            )
            reduced_contrast_ids = tuple(
                contrast_id
                for contrast_id in original_profile["contrast_ids"]
                if strategy_config.STRATEGY_COMPARE_CONTRASTS[contrast_id].get("left")
                != switch_target_arm_id
                and strategy_config.STRATEGY_COMPARE_CONTRASTS[contrast_id].get("right")
                != switch_target_arm_id
            )
            strategy_config.STRATEGY_COMPARE_PROFILES[profile_id]["arm_ids"] = reduced_arm_ids
            strategy_config.STRATEGY_COMPARE_PROFILES[profile_id]["contrast_ids"] = reduced_contrast_ids
            reduced_settings = strategy_config.get_strategy_comparison_settings(profile_id)
        finally:
            strategy_config.STRATEGY_COMPARE_PROFILES[profile_id].clear()
            strategy_config.STRATEGY_COMPARE_PROFILES[profile_id].update(original_profile)
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
    summary["app_path"] = "apps/research.py"
    summary["enabled_arms"] = [arm.arm_id for arm in settings.enabled_arms]
    summary["preparation"] = settings.preparation.as_dict()
    return results, summary

def validate_breakout_quality_stale_score_membership_guard_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STALE_SCORE_MEMBERSHIP_GUARD"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config.strategy_compare import get_strategy_comparison_settings
    from core.exact_accounting import build_buy_ledger_from_price
    from core.portfolio_entries import (
        reorder_candidates_for_resource_aware_quality,
        select_resource_aware_action_candidates,
    )
    from core.strategy_params import V16StrategyParams

    settings = get_strategy_comparison_settings()
    c25 = settings.arms.get("C25")
    c26 = settings.arms.get("C26")
    contrast_c26_c25 = settings.contrasts.get("C26-C25")
    contrast_c26_c23 = settings.contrasts.get("C26-C23")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "c26_registry_identity_is_c25_plus_selection_frozen_22d_stale_score_guard",
        True,
        c25 is not None
        and c26 is not None
        and c25.param_source == c26.param_source == "selection_min_roos"
        and c25.rule_policy == c26.rule_policy == "all_off"
        and c25.dl_id == c26.dl_id == "CONT12B_PIT"
        and c25.dl_runtime_mode == "resource-aware-continuous-max-dl-feasible-ascent"
        and c26.dl_runtime_mode == "resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard"
        and dict(c26.dl_runtime_options or {}) == {
            "stale_score_membership_guard_max_age_days": 22,
        }
        and contrast_c26_c25 is not None
        and contrast_c26_c25.left == "C26"
        and contrast_c26_c25.right == "C25"
        and contrast_c26_c23 is not None
        and contrast_c26_c23.left == "C26"
        and contrast_c26_c23.right == "C23",
    )

    resource_params = V16StrategyParams()
    resource_params.use_breakout_quality_ranking = True
    resource_params.breakout_quality_score_threshold = 0.5
    seed = (
        ("F0", 200.0, 153, 0.899),
        ("F1", 500.0, 249, 0.312),
        ("F2", 1000.0, 233, 0.346),
        ("F3", 1000.0, 162, 0.685),
        ("F4", 100.0, 334, 0.012),
        ("F5", 100.0, 326, 0.929),
        ("F6", 1000.0, 51, 0.706),
        ("F7", 300.0, 283, 0.585),
    )

    def candidate(raw, *, policy, score_date):
        ticker, price, qty, score = raw
        cost_milli = build_buy_ledger_from_price(
            price, qty, resource_params
        )["net_buy_total_milli"]
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
            "trade_date": "2024-01-10",
            "candidate_date": "2024-01-10",
            "signal_date": score_date,
            "breakout_quality_score_date": score_date,
            "breakout_quality_ranking_options": {
                "stale_score_membership_guard_max_age_days": 22,
            },
        }

    c25_rows = [
        candidate(
            raw,
            policy="resource-aware-continuous-max-dl-feasible-ascent",
            score_date="2024-01-09",
        )
        for raw in seed
    ]
    c26_fresh_rows = [
        candidate(
            raw,
            policy="resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
            score_date="2024-01-09",
        )
        for raw in seed
    ]
    c26_stale_rows = [
        candidate(
            raw,
            policy="resource-aware-continuous-max-dl-feasible-ascent-stale-score-guard",
            score_date=("2023-12-01" if raw[0] == "F3" else "2024-01-09"),
        )
        for raw in seed
    ]

    def run(rows):
        order, diag = reorder_candidates_for_resource_aware_quality(
            rows,
            available_cash=400_000.0,
            sizing_equity=2_000_000.0,
            pre_market_occupied=6,
            max_positions=10,
            params=resource_params,
        )
        action = select_resource_aware_action_candidates(order, diag)
        return order, action, diag

    c25_order, c25_action, c25_diag = run(c25_rows)
    fresh_order, fresh_action, fresh_diag = run(c26_fresh_rows)
    stale_order, stale_action, stale_diag = run(c26_stale_rows)

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "fresh_scores_preserve_c25_feasible_ascent_membership_exactly",
        True,
        [row["ticker"] for row in c25_action] == ["F2", "F3"]
        and [row["ticker"] for row in fresh_action]
        == [row["ticker"] for row in c25_action]
        and not fresh_diag["stale_score_guard_triggered"]
        and fresh_diag["stale_score_candidate_count"] == 0
        and fresh_diag["selected_count"] == c25_diag["selected_count"]
        and fresh_diag["reserved_cost_milli"] >= fresh_diag["baseline_reserved_cost_milli"],
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "stale_score_blocks_dl_membership_change_without_expiring_candidate_or_breaking_k_r0",
        True,
        [row["ticker"] for row in stale_action] == ["F1", "F2"]
        and "F3" in [row["ticker"] for row in stale_order]
        and stale_diag["stale_score_membership_guard_enabled"]
        and stale_diag["stale_score_membership_guard_max_age_days"] == 22
        and stale_diag["stale_score_candidate_count"] == 1
        and stale_diag["stale_score_guard_triggered"]
        and stale_diag["stale_score_guard_blocked_swaps"] > 0
        and stale_diag["selected_count"] == stale_diag["baseline_selected_count"]
        and stale_diag["reserved_cost_milli"] >= stale_diag["baseline_reserved_cost_milli"],
    )

    core_source = (
        Path(__file__).resolve().parents[2] / "core" / "portfolio_entry_selection_max_dl.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "stale_guard_core_is_runtime_option_driven_without_scientific_arm_id",
        True,
        "C26" not in core_source
        and "stale_score_membership_guard_max_age_days" in core_source,
    )

    summary.update({
        "cutoff_calendar_days": 22,
        "fresh_action": [row["ticker"] for row in fresh_action],
        "stale_action": [row["ticker"] for row in stale_action],
        "stale_candidates_retained": "F3" in [row["ticker"] for row in stale_order],
    })
    return results, summary

def validate_breakout_quality_daily_pit_strategy_runtime_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_DAILY_PIT_STRATEGY_RUNTIME"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr12b_keeps_event_score_date_while_mr13a_refreshes_latest_completed_information_date",
        ["2020-01-02", "2020-01-09"], lookup_dates,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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

    c1, c3, c20, c29, c30, c31 = (
        settings.arms["C1"], settings.arms["C3"], settings.arms["C20"],
        settings.arms["C29"], settings.arms["C30"], settings.arms["C31"],
    )
    forward_source = settings.dl_sources["CONT13A"]
    active_contrasts = tuple(
        item.contrast_id for item in settings.enabled_contrasts
    )
    configured_forward_arms = tuple(
        strategy_config.STRATEGY_COMPARE_PROFILES["forward_oos"]["arm_ids"]
    )
    configured_forward_contrasts = tuple(
        strategy_config.STRATEGY_COMPARE_PROFILES["forward_oos"]["contrast_ids"]
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_profile_honors_config_and_freezes_min_mr12b_mr13a_runtime_identity",
        True,
        {arm.arm_id for arm in settings.enabled_arms} == set(configured_forward_arms)
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_common_period_remains_auto_resolved_from_enabled_dl_sources",
        True,
        c1.param_source == "full_roos" and not c1.dl_enabled
        and c3.param_source == "min_roos" and not c3.dl_enabled
        and settings.start_date is None and settings.end_date is None,
    )

    project_root = Path(__file__).resolve().parents[2]
    strategy_diagnostics_source = (
        project_root / "filters" / "breakout_quality" / "strategy_compare_diagnostics.py"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        project_root / "services" / "breakout_quality" /
        "continuous_ranker_pipeline.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "training_and_strategy_diagnostics_share_domain_layer_profile_sample_provider",
        True,
        "load_profile_continuous_ranker_data" in strategy_diagnostics_source
        and "load_profile_continuous_ranker_data" in pipeline_source,
    )

    summary["profile"] = DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    summary["stage"] = "selection_strategy_translation"
    summary["future_target_used_for_score_presence"] = False
    return results, summary


def validate_breakout_quality_mr13e_strategy_source_gate_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_MR13E_STRATEGY_SOURCE_GATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config import strategy_compare as strategy_config

    selection = strategy_config.get_strategy_comparison_settings("selection_pit")
    forward = strategy_config.get_strategy_comparison_settings("forward_oos")
    selection_ids = tuple(arm.arm_id for arm in selection.enabled_arms)
    forward_ids = tuple(arm.arm_id for arm in forward.enabled_arms)

    c25, c28, c35 = (selection.arms[key] for key in ("C25", "C28", "C35"))
    c20, c29, c36 = (forward.arms[key] for key in ("C20", "C29", "C36"))
    pit_source = selection.dl_sources["CONT13E_PIT"]
    oos_source = forward.dl_sources["CONT13E"]

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_strategy_gate_sources_bind_only_to_mr13e_daily_profile",
        (
            DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "selection_point_in_time",
            DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "continuous_ranker_oos",
        ),
        (
            pit_source.experiment_profile,
            pit_source.score_source,
            oos_source.experiment_profile,
            oos_source.score_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_selection_arm_is_source_only_against_mr12b_and_mr13a",
        True,
        (
            (c35.param_source, c35.rule_policy, c35.dl_runtime_mode)
            == (c25.param_source, c25.rule_policy, c25.dl_runtime_mode)
            == (c28.param_source, c28.rule_policy, c28.dl_runtime_mode)
            and (c25.dl_id, c28.dl_id, c35.dl_id)
            == ("CONT12B_PIT", "CONT13A_PIT", "CONT13E_PIT")
            and c35.robustness_role == "off"
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_forward_arm_is_source_only_against_mr12b_and_mr13a",
        True,
        (
            (c36.param_source, c36.rule_policy, c36.dl_runtime_mode)
            == (c20.param_source, c20.rule_policy, c20.dl_runtime_mode)
            == (c29.param_source, c29.rule_policy, c29.dl_runtime_mode)
            and (c20.dl_id, c29.dl_id, c36.dl_id)
            == ("CONT12B", "CONT13A", "CONT13E")
            and c36.robustness_role == "off"
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_single_seed_gate_is_enabled_in_both_profiles_with_direct_source_contrasts",
        True,
        (
            selection_ids == ("C32", "C23", "C25", "C28", "C35")
            and forward_ids == ("C1", "C3", "C20", "C29", "C36")
            and {"C35-C25", "C35-C28"}.issubset(
                {item.contrast_id for item in selection.enabled_contrasts}
            )
            and {"C36-C20", "C36-C29"}.issubset(
                {item.contrast_id for item in forward.enabled_contrasts}
            )
        ),
    )

    selection_robust = strategy_config.get_strategy_multi_seed_robustness_settings(
        "selection_pit"
    )
    forward_robust = strategy_config.get_strategy_multi_seed_robustness_settings(
        "forward_oos"
    )
    selection_robust_profile = strategy_config.get_strategy_comparison_settings(
        selection_robust.profile_id
    )
    forward_robust_profile = strategy_config.get_strategy_comparison_settings(
        forward_robust.profile_id
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_does_not_enter_multi_seed_before_single_seed_strategy_gate",
        (("C25", "C28"), ("C20", "C29")),
        (
            tuple(
                arm.arm_id for arm in selection_robust_profile.enabled_arms
                if arm.robustness_role == "stochastic"
            ),
            tuple(
                arm.arm_id for arm in forward_robust_profile.enabled_arms
                if arm.robustness_role == "stochastic"
            ),
        ),
    )

    summary["selection_arm"] = "C35"
    summary["forward_arm"] = "C36"
    summary["runtime_source"] = "DL-CONT13E"
    summary["pit_source"] = "DL-CONT13E-PIT"
    return results, summary
