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
    preparation_status_path = project_root / "filters" / "breakout_quality" / "strategy_compare_preparation_status.py"
    dl_artifacts_path = project_root / "filters" / "breakout_quality" / "strategy_compare_dl_artifacts.py"
    reuse_path = project_root / "filters" / "breakout_quality" / "strategy_compare_reuse.py"
    runtime_path = project_root / "filters" / "breakout_quality" / "strategy_compare_runtime.py"
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

    from services.breakout_quality import continuous_target_builder, continuous_target_metrics

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_build_logic_is_owned_by_service_layer_not_audit_namespace",
        True,
        bool(
            callable(continuous_target_builder.build_strategy_aligned_target_artifacts)
            and callable(continuous_target_builder.build_strategy_aligned_no_time_target_artifacts)
            and callable(continuous_target_metrics.distribution_metrics)
            and not (project_root / "tools" / "audit" / "breakout_quality" / "continuous_target.py").exists()
            and not (project_root / "tools" / "audit" / "breakout_quality" / "no_time_continuous_target.py").exists()
        ),
    )

    config_source = config_path.read_text(encoding="utf-8")
    app_source = app_path.read_text(encoding="utf-8")
    model_app_source = model_app_path.read_text(encoding="utf-8")
    orchestration_source = orchestration_path.read_text(encoding="utf-8")
    preparation_source = (
        preparation_path.read_text(encoding="utf-8")
        + "\n"
        + preparation_status_path.read_text(encoding="utf-8")
        + "\n"
        + dl_artifacts_path.read_text(encoding="utf-8")
    )
    reuse_source = reuse_path.read_text(encoding="utf-8")
    runtime_source = runtime_path.read_text(encoding="utf-8")
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
        arm for arm in settings.enabled_arms if arm.dl_enabled
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
    menu_profiles = strategy_config.get_strategy_comparison_menu_profiles()
    menu_labels = tuple(str(item["label"]) for item in menu_profiles)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_main_menu_exposes_only_generic_config_selected_stages",
        True,
        tuple(item["profile_id"] for item in menu_profiles)
        == tuple(strategy_config.STRATEGY_COMPARE_MENU_PROFILE_IDS)
        and tuple(strategy_config.STRATEGY_COMPARE_MENU_PROFILE_IDS)
        == ("extending_window_rolling_fast", "extending_window_rolling")
        and len(menu_labels) == 2
        and "Fast Test" in menu_labels[0]
        and "Overnight Test" in menu_labels[1]
        and all("MR-" not in label for label in menu_labels),
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
    active_dl_ids.update(
        str(dict(strategy_config.STRATEGY_COMPARE_ARMS[arm_id].get("dl_runtime_options") or {}).get("safety_dl_id") or "")
        for arm_id in active_arm_ids
        if dict(strategy_config.STRATEGY_COMPARE_ARMS[arm_id].get("dl_runtime_options") or {}).get("safety_dl_id")
    )
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
        and "from services.breakout_quality.point_in_time_scores import (" not in preparation_source
        and "from services.breakout_quality.point_in_time_audit import (" not in preparation_source
        and "build_selection_point_in_time_scores" not in preparation_source
        and "audit_selection_point_in_time_scores" not in preparation_source
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
            "get_strategy_comparison_profiles", "查看目前Framework設定與工件狀態",
        ))
        and "C1" not in app_source and "TP1" not in app_source
        and '"strategy-compare"' not in model_app_source
        and "run_strategy_comparison" not in model_app_source,
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
    current_modes = strategy_config.get_strategy_rolling_test_modes()
    current_comparisons = tuple(
        strategy_config.get_strategy_comparison_settings(str(mode["profile_id"]))
        for mode in current_modes
    )
    current_required_pit_sources = [
        source
        for comparison in current_comparisons
        for source in comparison.dl_sources.values()
        if source.dl_id in {
            str(arm.dl_id)
            for arm in comparison.enabled_arms
            if arm.dl_enabled and arm.dl_id
        }
        and source.score_source == "selection_point_in_time"
    ]
    model_prepare_source = model_app_source.split(
        "def _prepare_strategy_compare_model_artifacts", 1
    )[1].split("def _interactive_model_research", 1)[0]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "rolling_compare_is_consumer_only_and_model_work_type_owns_pit_build_audit_and_missing_fold_training",
        True,
        bool(current_required_pit_sources)
        and all(source.threshold is None for source in current_required_pit_sources)
        and all(
            source.forward_scores_builder is not None
            and source.forward_scores_builder.builder_type == "selection_pit_from_existing_folds"
            for source in current_required_pit_sources
        )
        and "load_selection_point_in_time_ranking_contract" in preparation_source
        and "Strategy Compare只消費既有PIT" in preparation_source
        and "不建立、不重建也不執行PIT Model Gate" in preparation_source
        and "準備策略比較所需模型工件" in preparation_source
        and "checkpoint_only=True" not in preparation_source
        and "audit_selection_point_in_time_scores" not in preparation_source
        and "build_selection_point_in_time_scores" in model_prepare_source
        and "audit_selection_point_in_time_scores" in model_prepare_source
        and "resume=True" in model_prepare_source
        and "point_in_time_fold_months" in model_prepare_source
        and "point_in_time_fold_anchor_date" in model_prepare_source
        and "point_in_time_dirname" in model_prepare_source
        and "缺少／不相容fold才補訓" in model_prepare_source
        and "Strategy Compare不得因此訓練模型" in (
            project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
        ).read_text(encoding="utf-8")
        and any(source.artifact_contract is not None for source in settings.parameter_sources.values()),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "model_work_type_prepares_all_configured_current_rolling_sources_without_strategy_compare_training",
        True,
        "_strategy_compare_required_model_sources" in model_app_source
        and "get_strategy_comparison_menu_profiles" in model_app_source
        and "get_strategy_rolling_test_modes" in model_app_source
        and "SCORE_SOURCE_SELECTION_POINT_IN_TIME" in model_prepare_source
        and "build_selection_point_in_time_scores" in model_prepare_source
        and "audit_selection_point_in_time_scores" in model_prepare_source
        and all(
            comparison.profile_id == str(mode["profile_id"])
            and all(
                source.score_source == "selection_point_in_time"
                and int(source.point_in_time_fold_months or 0) == int(mode["fold_months"])
                and (source.point_in_time_fold_anchor_date or None) == (mode.get("fold_anchor_date") or None)
                for source in comparison.dl_sources.values()
                if source.dl_id in {
                    str(arm.dl_id)
                    for arm in comparison.enabled_arms
                    if arm.dl_enabled and arm.dl_id
                }
            )
            for mode, comparison in zip(current_modes, current_comparisons)
        )
        and "train-continuous-ranker" not in preparation_source
        and "準備策略比較所需模型工件" in preparation_source,
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
        and "def _comparison_runs_roots(" in reuse_source
        and "def _comparison_runs_roots(" not in orchestration_source,
    )

    display_alignment_groups = {}
    for profile_id, raw_profile in strategy_config.STRATEGY_COMPARE_PROFILES.items():
        group = str(raw_profile.get("display_alignment_group") or "").strip()
        if not group:
            continue
        settings_for_alignment = strategy_config.get_strategy_comparison_settings(profile_id)
        alignment_ids = tuple(raw_profile.get("display_alignment_arm_ids") or raw_profile["arm_ids"])
        display_alignment_groups.setdefault(group, []).append(
            tuple(settings_for_alignment.arms[arm_id].name for arm_id in alignment_ids)
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
    configured_robustness_ids = set(strategy_config.STRATEGY_COMPARE_MULTI_SEED_ROBUSTNESS_PROFILES)
    enabled_robustness_ids = {item["robustness_id"] for item in robustness_profiles}
    rolling_modes = strategy_config.get_strategy_rolling_test_modes()
    expected_enabled_robustness_ids = {str(mode["robustness_id"]) for mode in rolling_modes}
    robustness_mode_contracts = []
    for mode in rolling_modes:
        mode_settings = strategy_config.get_strategy_multi_seed_robustness_settings(str(mode["robustness_id"]))
        mode_profile = strategy_config.get_strategy_comparison_settings(mode_settings.profile_id)
        fixed = tuple(mode_profile.arms[arm_id] for arm_id in mode_settings.fixed_arm_ids)
        stochastic = tuple(mode_profile.arms[arm_id] for arm_id in mode_settings.stochastic_arm_ids)
        reference_specs = dict(mode_settings.romd_reference_baselines)
        reference_matches = {
            key: tuple(
                arm for arm in fixed
                if arm.param_source == spec["param_source"] and arm.rule_policy == spec["rule_policy"]
            )
            for key, spec in reference_specs.items()
        }
        robustness_mode_contracts.append(
            mode_settings.profile_id == str(mode["profile_id"])
            and mode_settings.seed_count >= 2
            and 1 <= mode_settings.gpu_train_workers <= 2
            and mode_settings.cpu_replay_workers >= 1
            and bool(fixed)
            and bool(stochastic)
            and all(not arm.dl_enabled for arm in fixed)
            and all(
                arm.dl_enabled
                and mode_profile.dl_sources[str(arm.dl_id)].score_source == "selection_point_in_time"
                for arm in stochastic
            )
            and set(reference_matches) == {"min"}
            and all(len(matches) == 1 for matches in reference_matches.values())
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_robustness_matrix_is_rolling_mode_config_driven_without_mutating_single_seed_arm_identity",
        True,
        robustness_path.is_file()
        and expected_enabled_robustness_ids.issubset(configured_robustness_ids)
        and enabled_robustness_ids == expected_enabled_robustness_ids
        and robustness_settings.robustness_id in expected_enabled_robustness_ids
        and all(robustness_mode_contracts)
        and "romd_reference_baselines" in config_source
        and "fixed_arm_ids" in config_source
        and "stochastic_arm_ids" in config_source
        and "MULTI_SEED_ROBUSTNESS_ARM_IDS" not in config_source
        and all(token not in robustness_source for token in ("\"C20\"", "\"C29\"", "\"MR-12B\"", "\"MR-13A\"")),
    )

    # Downstream synthetic cases exercise the configured default robustness mode.
    robustness_profile = strategy_config.get_strategy_comparison_settings(
        robustness_settings.profile_id
    )
    robustness_fixed = tuple(
        robustness_profile.arms[arm_id] for arm_id in robustness_settings.fixed_arm_ids
    )
    robustness_stochastic = tuple(
        robustness_profile.arms[arm_id] for arm_id in robustness_settings.stochastic_arm_ids
    )
    default_reference_specs = dict(robustness_settings.romd_reference_baselines)
    reference_matches = {
        key: tuple(
            arm for arm in robustness_fixed
            if arm.param_source == spec["param_source"]
            and arm.rule_policy == spec["rule_policy"]
        )
        for key, spec in default_reference_specs.items()
    }

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

    shared_source_arm = robustness_profile.arms["C59"]
    shared_dl_stochastic = (
        replace(shared_source_arm, arm_id="SYN_SHARED_A", enabled=True),
        replace(shared_source_arm, arm_id="SYN_SHARED_B", enabled=True),
    )
    source_groups = robustness_module._training_source_groups(shared_dl_stochastic, settings=robustness_profile)
    dedupe_seeds = (101, 202)
    dedupe_units = list(robustness_module._training_units(
        seeds=dedupe_seeds,
        stochastic_arms=shared_dl_stochastic,
        settings=robustness_profile,
        completed=set(),
        model_root=Path("/tmp/synthetic_multi_seed_models"),
        run_root=Path("/tmp/synthetic_multi_seed_run"),
        comparison_start="2021-01-01",
        comparison_end="2025-12-31",
        reuse_completed=True,
    ))
    group_arms = {
        dl_id: tuple(arm.arm_id for _arm_order, arm in entries)
        for dl_id, entries in source_groups
    }
    dedupe_unit_replays = {
        (str(unit["dl_id"]), int(unit["seed"])):
            tuple(arm.arm_id for _arm_order, arm in unit["replay_arms"])
        for unit in dedupe_units
    }
    shared_source_groups = {
        dl_id: arm_ids for dl_id, arm_ids in group_arms.items() if len(arm_ids) > 1
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_training_deduplicates_same_dl_source_per_seed_then_fans_out_to_multiple_runtime_arms",
        True,
        bool(group_arms)
        and len(dedupe_units) == len(source_groups) * len(dedupe_seeds)
        and sum(len(unit["replay_arms"]) for unit in dedupe_units)
            == len(shared_dl_stochastic) * len(dedupe_seeds)
        and all(
            dedupe_unit_replays[(dl_id, seed)] == arm_ids
            for dl_id, arm_ids in group_arms.items()
            for seed in dedupe_seeds
        )
        and all(
            sum(1 for unit in dedupe_units if str(unit["dl_id"]) == dl_id and int(unit["seed"]) == seed) == 1
            for dl_id in group_arms
            for seed in dedupe_seeds
        )
        and "replay_arms" in robustness_source
        and "training_cleanup_targets" in robustness_source,
    )

    operational_groups = robustness_module._training_source_groups(
        robustness_stochastic, settings=robustness_profile
    )
    operational_units = list(robustness_module._training_units(
        seeds=dedupe_seeds, stochastic_arms=robustness_stochastic, settings=robustness_profile,
        completed=set(), model_root=Path("/tmp/extending_window_rolling_models"),
        run_root=Path("/tmp/extending_window_rolling_run"),
        comparison_start="2021-01-01", comparison_end="2025-12-31", reuse_completed=True,
    ))
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_operational_dual_model_profile_trains_both_same_seed_sources_before_single_strategy_replay",
        True,
        tuple(arm.arm_id for arm in robustness_stochastic) == ("C60",)
        and tuple((str(item["left"]), str(item["right"])) for item in robustness_settings.paired_contrasts) == ()
        and len(operational_groups) == 2
        and len(operational_units) == 2 * len(dedupe_seeds)
        and sum(len(unit["replay_arms"]) for unit in operational_units) == 2 * len(dedupe_seeds)
        and "queue_replay_if_ready" in robustness_source
        and "trained_artifacts" in robustness_source
        and "_arm_training_dl_ids" in robustness_source,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_activation_is_decoupled_from_single_seed_arm_roles_and_uses_only_current_operational_profile_arms",
        True,
        strategy_config.STRATEGY_COMPARE_SCHEMA_VERSION >= 42
        and all(
            robustness_profile.arms[arm_id].robustness_role == "off"
            for arm_id in robustness_settings.stochastic_arm_ids
        )
        and set(robustness_settings.fixed_arm_ids + robustness_settings.stochastic_arm_ids)
            <= {arm.arm_id for arm in robustness_profile.enabled_arms},
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
    full_reference_matches = reference_matches.get("full", ())
    full_reference_arm = full_reference_matches[0] if full_reference_matches else None
    fixed_results = {}
    for fixed_order, arm in enumerate(robustness_fixed, start=1):
        if full_reference_arm is not None and arm.arm_id == full_reference_arm.arm_id:
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
    configured_pairs = tuple(robustness_settings.paired_contrasts)
    paired_by_id = {
        str(item["contrast_id"]): item
        for item in synthetic_summary.get("paired_comparisons") or []
    }
    arm_order_by_id = {
        arm.arm_id: arm_order
        for arm_order, arm in enumerate(robustness_stochastic, start=1)
    }
    paired_comparisons_ok = (
        set(paired_by_id) == {str(item["contrast_id"]) for item in configured_pairs}
        and len(paired_by_id) == len(configured_pairs)
    )
    for spec in configured_pairs:
        contrast_id = str(spec["contrast_id"])
        left_id = str(spec["left"])
        right_id = str(spec["right"])
        item = paired_by_id.get(contrast_id)
        if item is None:
            paired_comparisons_ok = False
            continue
        expected_romd_delta = 3.0 * (
            arm_order_by_id[right_id] - arm_order_by_id[left_id]
        )
        expected_direct_delta = expected_romd_delta * 2.0
        romd_same = item.get("romd_same_seed")
        direct_same = item.get("direct_selection_r_same_seed")
        translation = item.get("selection_r_to_strategy")
        distribution = item.get("romd_distribution")
        yearly_pair = list(item.get("yearly_same_seed") or [])
        expected_right_wins = 2 if expected_romd_delta > 0 else 0
        expected_left_wins = 2 if expected_romd_delta < 0 else 0
        paired_comparisons_ok = paired_comparisons_ok and (
            item.get("left_arm_id") == left_id
            and item.get("right_arm_id") == right_id
            and isinstance(romd_same, dict)
            and romd_same.get("n") == 2
            and romd_same.get("right_gt_left_count") == expected_right_wins
            and romd_same.get("left_gt_right_count") == expected_left_wins
            and romd_same.get("tie_count") == (2 if math.isclose(expected_romd_delta, 0.0) else 0)
            and math.isclose(float(romd_same.get("right_minus_left_mean")), expected_romd_delta)
            and math.isclose(float(romd_same.get("right_minus_left_median")), expected_romd_delta)
            and isinstance(direct_same, dict)
            and direct_same.get("n") == 2
            and math.isclose(float(direct_same.get("right_minus_left_mean")), expected_direct_delta)
            and isinstance(translation, dict)
            and translation.get("n") == 2
            and len(translation.get("seed_rows") or []) == 2
            and [row["seed_index"] for row in translation.get("seed_rows") or []] == [1, 2]
            and isinstance(distribution, dict)
            and distribution.get("pair_count") == 4
            and len(yearly_pair) == 2
            and all(row.get("n") == 2 for row in yearly_pair)
        )
    legacy_pair_fields_ok = all(
        synthetic_summary.get(key) is None
        for key in (
            "romd_distribution_comparison",
            "romd_same_seed_comparison",
            "direct_selection_r_same_seed_comparison",
            "selection_r_to_strategy_same_seed_translation",
        )
    ) if len(configured_pairs) != 1 else True

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_report_uses_mean_for_all_strategy_metrics_and_full_romd_distribution_with_fixed_baselines",
        True,
        math.isclose(mean_by_arm[min_reference_arm.arm_id]["return_over_max_drawdown"], 6.0)
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
            and romd_by_arm[arm.arm_id]["beats_full_count"] is None
            for arm in robustness_stochastic
        )
        and synthetic_summary["contract"]["romd_reference_baselines"]["min"]["arm_id"]
        == min_reference_arm.arm_id
        and "full" not in synthetic_summary["contract"]["romd_reference_baselines"]
        and paired_comparisons_ok
        and legacy_pair_fields_ok
        and len(synthetic_summary["yearly_statistics"]) >= len(robustness_fixed) * 2 + len(robustness_stochastic) * 2
        and len(synthetic_summary["yearly_same_seed_comparison"]) == len(configured_pairs) * 2
        and math.isclose(fixed_yearly_side[0]["return_pct"], 1.25)
        and math.isclose(stochastic_yearly_side[0]["return_pct"], 9.75)
        and robustness_source.count('result_side="no_filter"') >= 2
        and "_load_direct_selection_r(" in robustness_source
        and "active_trades_filename=runtime_spec[\"active_trades_filename\"]" in robustness_source,
    )

    rendered_common_report = robustness_module.render_multi_seed_robustness_report(synthetic_summary)
    historical_summary = dict(synthetic_summary)
    historical_summary["schema_version"] = 8
    historical_summary.pop("common_strategy_report", None)
    historical_contract = dict(historical_summary["contract"])
    historical_contract["report_schema_version"] = 8
    historical_contract["resolved_seeds"] = [101, 202]
    historical_contract["seed_count"] = 2
    historical_summary["contract"] = historical_contract
    with patch.object(
        robustness_module, "_train_one_unit", side_effect=AssertionError("report refresh must not train"),
    ), patch.object(
        robustness_module, "_replay_one_unit", side_effect=AssertionError("report refresh must not replay"),
    ), patch.object(
        robustness_module, "_reusable_fixed_scenario_metrics", return_value={},
    ):
        upgraded_historical_summary = robustness_module._upgrade_derived_report_summary(
            historical_summary,
            seed_frame=synthetic_frame,
            seed_yearly_frame=synthetic_yearly_frame,
            run_root=None,
        )
    upgraded_common_report = robustness_module.render_multi_seed_robustness_report(
        upgraded_historical_summary
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_report_reuses_canonical_strategy_sections_and_upgrades_historical_results_without_train_or_replay",
        True,
        synthetic_summary.get("schema_version") == robustness_module.ROBUSTNESS_SCHEMA_VERSION
        and bool(synthetic_summary.get("common_strategy_report"))
        and upgraded_historical_summary.get("schema_version") == robustness_module.ROBUSTNESS_SCHEMA_VERSION
        and bool(upgraded_historical_summary.get("common_strategy_report"))
        and all(
            section in rendered_common_report and section in upgraded_common_report
            for section in (
                "策略績效比較",
                "1. 核心策略結果",
                "2. R 預測／轉化",
                "3. 資金／執行",
                "4. 年度結果",
                "5. RoMD完整統計",
            )
        )
        and "## " not in rendered_common_report
        and "## " not in upgraded_common_report
        and "render_strategy_aggregate_report(" in robustness_source
        and "_print_report_tables(summary)" in robustness_source,
    )

    rendered_plain_report = robustness_module.render_multi_seed_robustness_report(
        synthetic_summary, target="plain"
    )
    multi_seed_extension = rendered_common_report.split("5. RoMD完整統計", 1)[-1]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_unique_sections_reuse_project_color_semantics_and_plain_fallback",
        True,
        "#188038" in multi_seed_extension
        and "#C62828" in multi_seed_extension
        and "<span" not in rendered_plain_report
        and "ROBUSTNESS_ROMD_DISTRIBUTION_METRICS" in robustness_source
        and "ROBUSTNESS_SEED_DELTA_METRICS" in robustness_source
        and "best_worst_signals" in robustness_source
        and "styled_signal" in robustness_source,
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

    from filters.breakout_quality import ranking_score_store as ranking_store_module

    with tempfile.TemporaryDirectory() as tmp:
        override_project_root = Path(tmp)
        override_dir = override_project_root / "isolated_fast_60m"
        override_dir.mkdir(parents=True, exist_ok=True)
        override_score = override_dir / "selection_point_in_time_scores.csv"
        override_coverage = override_dir / "selection_point_in_time_coverage.csv"
        override_manifest = override_dir / "selection_point_in_time_manifest.json"
        override_audit = override_dir / "selection_point_in_time_audit.json"
        override_score.write_text("synthetic-score\n", encoding="utf-8")
        override_coverage.write_text("synthetic-coverage\n", encoding="utf-8")
        override_profile_id = "daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise"
        override_profile = get_breakout_quality_experiment_profile(override_profile_id)
        override_target_id = str(override_profile.continuous_target_id or "synthetic_target")
        override_period = {"start": "2016-01-01", "end": "2025-12-31"}
        override_manifest.write_text(
            json.dumps({
                "filter_id": "breakout_quality_v1",
                "model_architecture": "inception_time_v1",
                "experiment_profile": override_profile_id,
                "training_sample_scope": override_profile.training_sample_scope,
                "status": "BUILT",
                "seed": 42,
                "continuous_target_id": override_target_id,
                "score_period": override_period,
                "coverage": {"scored_group_count": 1},
                "artifacts": {"scores": build_file_manifest(override_score)},
            }),
            encoding="utf-8",
        )
        override_audit.write_text(
            json.dumps({
                "schema_version": 3,
                "status": "RESULT_AVAILABLE",
                "filter_id": "breakout_quality_v1",
                "model_architecture": "inception_time_v1",
                "experiment_profile": override_profile_id,
                "workflow": {"training_sample_scope": override_profile.training_sample_scope},
                "continuous_target_id": override_target_id,
                "score_period": override_period,
                "score_group_count": 1,
                "source_artifacts": {
                    "point_in_time_manifest": {
                        "path": str(override_manifest.resolve()),
                        **build_file_manifest(override_manifest),
                    },
                    "point_in_time_scores": {
                        "path": str(override_score.resolve()),
                        **build_file_manifest(override_score),
                    },
                    "point_in_time_coverage": {
                        "path": str(override_coverage.resolve()),
                        **build_file_manifest(override_coverage),
                    },
                    "continuous_target_manifest": {},
                },
            }),
            encoding="utf-8",
        )
        ranking_store_module.load_selection_point_in_time_ranking_contract.cache_clear()
        with patch.object(
            ranking_store_module, "_validate_score_eligibility_contract", return_value=None
        ), patch.object(
            ranking_store_module, "_validate_embedded_target_source_artifact", return_value=None
        ), patch.object(
            ranking_store_module,
            "derive_point_in_time_model_validation_gate",
            return_value={"status": "PASS", "checks": {}},
        ):
            override_contract = ranking_store_module.load_selection_point_in_time_ranking_contract(
                str(override_project_root),
                "breakout_quality_v1",
                "inception_time_v1",
                override_profile_id,
                require_model_validation_pass=False,
                point_in_time_dir_override=override_dir,
            )
        ranking_store_module.load_selection_point_in_time_ranking_contract.cache_clear()

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_override_keeps_score_manifest_coverage_and_audit_in_same_mode_namespace",
        True,
        override_contract.score_path == override_score.resolve()
        and override_contract.manifest_path == override_manifest.resolve()
        and override_contract.audit_path == override_audit.resolve()
        and override_contract.available_from == "2016-01-01"
        and override_contract.available_through == "2025-12-31",
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
        and '"execution_start": str(artifacts.get("score_execution_start") or "")' in robustness_source
        and 'continuous_score_execution_start_override=' in robustness_source
        and 'primary_override.get("execution_start") or job["score_execution_start"]' in robustness_source
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
        "strategy_compare_selection_pit_replay_uses_inspection_contract_and_pins_validated_paths_without_relaxing_global_runtime_loader",
        True,
        "require_model_validation_pass=False" in engine_source
        and '"score_path_override": str(pit_contract.score_path)' in engine_source
        and '"score_manifest_path_override": str(pit_contract.manifest_path)' in engine_source
        and "load_selection_point_in_time_ranking_contract(" in engine_source,
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
    if configured_required_sources:
        build_source = configured_required_sources[0]
        reuse_sources = configured_required_sources[1:]
        first_actions = [
            StrategyPreparationAction(
                action_id=f"param:{build_source}", artifact_key=f"param:{build_source}",
                action="BUILD", builder_type="synthetic_builder", description="build", path="models/build.json",
            )
        ]
        first_actions.extend(
            StrategyPreparationAction(
                action_id=f"param:{source_id}", artifact_key=f"param:{source_id}",
                action="REUSE", builder_type=None, description="reuse", path=f"models/{source_id}.json",
            )
            for source_id in reuse_sources
        )
        first_actions.append(dl_blocked)
        second_actions = [
            StrategyPreparationAction(
                action_id=f"param:{source_id}", artifact_key=f"param:{source_id}",
                action="REUSE", builder_type=None, description="reuse",
                path=("models/build.json" if source_id == build_source else f"models/{source_id}.json"),
            )
            for source_id in configured_required_sources
        ]
        second_actions.append(dl_blocked)
        first_plan = StrategyPreparationPlan(
            overall_status="BLOCKED", actions=tuple(first_actions),
        )
        second_plan = StrategyPreparationPlan(
            overall_status="BLOCKED", actions=tuple(second_actions),
        )
        prep_calls = []
        with patch.object(
            preparation_module, "_execute_preparation_action",
            side_effect=lambda **kwargs: prep_calls.append(kwargs["action"].artifact_key),
        ):
            prepared = preparation_module.prepare_strategy_parameter_artifacts(
                settings=fake_settings,
                status={"preparation_plan": first_plan},
                required_source_ids=tuple(configured_required_sources),
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

    configured_start, configured_end = robustness_module._comparison_period_from_upstream(
        robustness_profile, robustness_stochastic, {}
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_period_honors_explicit_current_profile_window_before_dataset_tail_fallback",
        True,
        configured_start == str(pd.Timestamp(str(robustness_profile.start_date)).date())
        and configured_end == str(pd.Timestamp(str(robustness_profile.end_date)).date()),
    )

    dataset_profile_mismatch_rejected = False
    fallback_profile = replace(robustness_profile, start_date=None, end_date=None)
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
                    fallback_profile, robustness_stochastic, {}
                )
            except RuntimeError:
                dataset_profile_mismatch_rejected = True
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "multi_seed_dataset_tail_fallback_rejects_dataset_profile_mismatch",
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
        operational_param_key = f"param:{robustness_fixed[0].param_source}"
        fingerprint_base = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
            artifact_identities={operational_param_key: {"sha256": "a"}},
        )["fingerprint"]
        fingerprint_period = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2026-01-31"},
            artifact_identities={operational_param_key: {"sha256": "a"}},
        )["fingerprint"]
        fingerprint_param = robustness_module.build_multi_seed_robustness_contract(
            comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
            artifact_identities={operational_param_key: {"sha256": "changed"}},
        )["fingerprint"]
        with patch.object(
            robustness_module, "BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE",
            float(robustness_module.BREAKOUT_QUALITY_DEFAULT_LEARNING_RATE) * 2.0,
        ):
            fingerprint_training = robustness_module.build_multi_seed_robustness_contract(
                comparison_period={"start": "2021-01-01", "end": "2025-12-22"},
                artifact_identities={operational_param_key: {"sha256": "a"}},
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
                artifact_identities={operational_param_key: {"sha256": "a"}},
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

    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_completed_pair_score_reuse_contract_checks,
    )
    append_completed_pair_score_reuse_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
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

    from filters.breakout_quality.strategy_compare_preparation_status import (
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

    blocked_render_plan = StrategyPreparationPlan.from_actions((
        StrategyPreparationAction(
            action_id="param:synthetic",
            artifact_key="param:synthetic",
            action="REBUILD",
            builder_type="synthetic",
            description="synthetic preparable action",
            path="models/synthetic.json",
        ),
        StrategyPreparationAction(
            action_id="dl:synthetic:blocker",
            artifact_key="dl:synthetic:blocker",
            action="BLOCKED",
            builder_type=None,
            description="synthetic upstream blocker",
            path="models/synthetic.blocked",
        ),
    ))
    blocked_plan_text = strategy_comparison_module.render_execution_plan(
        settings=settings,
        status={
            "preparation_plan": blocked_render_plan,
            "replay_cache": {},
            "config_fingerprint": "synthetic",
        },
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "blocked_execution_plan_marks_nonexecuted_build_run_and_report_rows_as_not_run",
        True,
        blocked_render_plan.overall_status == "BLOCKED"
        and "BLOCKED" in blocked_plan_text
        and blocked_plan_text.count("NOT_RUN")
        >= 1 + len(settings.enabled_arms) + len(settings.enabled_contrasts)
        and "REPORT" not in blocked_plan_text
        and "\nRUN" not in blocked_plan_text,
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "post_preparation_refresh_uses_canonical_resolved_plan_boundary",
        True,
        all(token in orchestration_source for token in (
            "resolved_plan = resolve_comparison_plan(",
            "ResolvedComparisonPlan.from_status(",
            "READY has already passed ResolvedComparisonPlan.validate_contract()",
        )),
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
    from core.portfolio_entry_selection_common import (
        _decorate_same_day_rank_residual_safety_scores,
    )
    from core.portfolio_entry_selection_max_dl import (
        build_max_dl_repair_mechanism_diagnostic,
        _excess_alpha_basket_quality_key,
        _max_dl_basket_quality_key,
        _score_capital_basket_quality_key,
        _max_dl_execution_order,
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


    def _resource_candidate_fixed(
        ticker, price, qty, score, policy, expected_r=None, expected_excess_r=None, safety_score=None
    ):
        cost_milli = build_buy_ledger_from_price(price, qty, resource_params)["net_buy_total_milli"]
        rank_payload = {"available": True, "score": score}
        if expected_r is not None:
            rank_payload.update({
                "expected_r_available": True,
                "expected_r": float(expected_r),
                "daily_score_percentile": float(score),
            })
        if expected_excess_r is not None:
            rank_payload.update({
                "expected_excess_r_available": True,
                "expected_excess_r": float(expected_excess_r),
                "daily_score_percentile": float(score),
            })
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
            "breakout_quality_safety_score": (None if safety_score is None else float(safety_score)),
            "breakout_quality_safety_score_available": bool(safety_score is not None),
            "breakout_quality_rank": rank_payload,
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


    expected_pnl_seed = (
        ("EP_A", 100.0, 600, 0.99, 1.80),
        ("EP_B", 100.0, 1500, 0.95, 1.60),
        ("EP_C", 100.0, 1600, 0.90, 1.40),
    )
    expected_pnl_score_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
            expected_r,
        )
        for ticker, price, qty, score, expected_r in expected_pnl_seed
    ]
    expected_pnl_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-expected-pnl-feasible-ascent",
            expected_r,
        )
        for ticker, price, qty, score, expected_r in expected_pnl_seed
    ]
    score_order, score_diag = reorder_candidates_for_resource_aware_quality(
        expected_pnl_score_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    pnl_order, pnl_diag = reorder_candidates_for_resource_aware_quality(
        expected_pnl_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    score_action = select_resource_aware_action_candidates(score_order, score_diag)
    pnl_action = select_resource_aware_action_candidates(pnl_order, pnl_diag)
    score_selected = _simulate_reserved_candidate_order(
        score_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=2, params=resource_params,
    )
    pnl_selected = _simulate_reserved_candidate_order(
        pnl_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=2, params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "expected_pnl_selector_changes_only_basket_objective_while_preserving_same_k_r0_and_prefers_larger_deployable_expected_profit",
        True,
        [row["ticker"] for row in score_action] == ["EP_A", "EP_B"]
        and [row["ticker"] for row in pnl_action] == ["EP_B", "EP_C"]
        and pnl_diag.get("basket_objective") == "expected_pnl"
        and score_diag.get("basket_objective") == "score"
        and pnl_diag["pre_market_order_limit"] == score_diag["pre_market_order_limit"] == 2
        and pnl_selected["selected_count"] == score_selected["selected_count"] == 2
        and pnl_selected["reserved_cost_milli"] >= score_diag["baseline_reserved_cost_milli"]
        and score_selected["reserved_cost_milli"] >= score_diag["baseline_reserved_cost_milli"],
    )

    excess_alpha_seed = (
        ("EA_A", 100.0, 600, 0.99, 0.30),
        ("EA_B", 100.0, 1500, 0.95, 0.20),
        ("EA_C", 100.0, 1600, 0.90, 0.19),
    )
    excess_score_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-max-dl-feasible-ascent",
        )
        for ticker, price, qty, score, _expected_excess_r in excess_alpha_seed
    ]
    excess_alpha_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-feasible-ascent",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in excess_alpha_seed
    ]
    excess_score_order, excess_score_diag = reorder_candidates_for_resource_aware_quality(
        excess_score_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    excess_order, excess_diag = reorder_candidates_for_resource_aware_quality(
        excess_alpha_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    excess_score_action = select_resource_aware_action_candidates(
        excess_score_order, excess_score_diag
    )
    excess_action = select_resource_aware_action_candidates(excess_order, excess_diag)
    excess_score_selected = _simulate_reserved_candidate_order(
        excess_score_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=2, params=resource_params,
    )
    excess_selected = _simulate_reserved_candidate_order(
        excess_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=2, params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "excess_alpha_selector_changes_only_basket_objective_while_preserving_same_k_r0_and_balances_relative_quality_with_deployable_risk",
        True,
        [row["ticker"] for row in excess_score_action] == ["EA_A", "EA_B"]
        and [row["ticker"] for row in excess_action] == ["EA_B", "EA_C"]
        and excess_diag.get("basket_objective") == "excess_alpha"
        and excess_score_diag.get("basket_objective") == "score"
        and excess_diag["pre_market_order_limit"] == excess_score_diag["pre_market_order_limit"] == 2
        and excess_selected["selected_count"] == excess_score_selected["selected_count"] == 2
        and excess_selected["reserved_cost_milli"] >= excess_score_diag["baseline_reserved_cost_milli"]
        and excess_score_selected["reserved_cost_milli"] >= excess_score_diag["baseline_reserved_cost_milli"],
    )

    no_r0_seed = (
        ("NR_H1", 100.0, 1500, 0.30, 0.02),
        ("NR_H2", 100.0, 1500, 0.20, 0.01),
        ("NR_A", 100.0, 600, 0.99, 0.50),
        ("NR_B", 100.0, 600, 0.98, 0.40),
    )
    with_r0_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-feasible-ascent",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in no_r0_seed
    ]
    no_r0_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-no-r0-feasible-ascent",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in no_r0_seed
    ]
    with_r0_order, with_r0_diag = reorder_candidates_for_resource_aware_quality(
        with_r0_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    no_r0_order, no_r0_diag = reorder_candidates_for_resource_aware_quality(
        no_r0_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    with_r0_action = select_resource_aware_action_candidates(with_r0_order, with_r0_diag)
    no_r0_action = select_resource_aware_action_candidates(no_r0_order, no_r0_diag)
    no_r0_selected = _simulate_reserved_candidate_order(
        no_r0_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "excess_alpha_no_r0_keeps_k_and_true_cash_feasibility_but_removes_r0_floor_and_r0_minimum_repair",
        True,
        [row["ticker"] for row in no_r0_action] == ["NR_A", "NR_B"]
        and [row["ticker"] for row in with_r0_action] != ["NR_A", "NR_B"]
        and no_r0_diag.get("basket_objective") == "excess_alpha"
        and no_r0_diag.get("resource_preservation_required") is False
        and int(no_r0_diag.get("max_dl_repair_steps", -1)) == 0
        and int(no_r0_diag.get("max_dl_repair_evaluations", -1)) == 0
        and no_r0_diag.get("pre_market_order_limit") == 2
        and no_r0_selected["selected_count"] == 2
        and no_r0_selected["reserved_cost_milli"] < int(no_r0_diag["baseline_reserved_cost_milli"]),
    )

    constrained_seed = (
        ("CO_H1", 100.0, 1400, 0.20, 0.03),
        ("CO_H2", 100.0, 1350, 0.19, 0.02),
        ("CO_A", 100.0, 700, 0.99, 0.45),
        ("CO_B", 100.0, 750, 0.98, 0.40),
        ("CO_C", 100.0, 900, 0.97, 0.28),
    )
    constrained_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in constrained_seed
    ]
    constrained_order, constrained_diag = reorder_candidates_for_resource_aware_quality(
        constrained_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    constrained_action = select_resource_aware_action_candidates(
        constrained_order, constrained_diag
    )
    constrained_result = _simulate_reserved_candidate_order(
        constrained_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    constrained_baseline = _simulate_reserved_candidate_order(
        constrained_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    constrained_base_rank = {id(row): idx for idx, row in enumerate(constrained_rows)}
    brute_best = None
    brute_best_key = None
    for combo in itertools.combinations(constrained_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=constrained_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if (
            combo_result["selected_count"] != 2
            or combo_result["reserved_cost_milli"] < constrained_baseline["reserved_cost_milli"]
        ):
            continue
        combo_key = _excess_alpha_basket_quality_key(
            combo_result, base_rank=constrained_base_rank
        )
        if brute_best_key is None or combo_key > brute_best_key:
            brute_best_key = combo_key
            brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "excess_alpha_constrained_solver_matches_exhaustive_canonical_k_r0_oracle_and_certifies_global_optimum",
        True,
        constrained_diag.get("basket_objective") == "excess_alpha"
        and constrained_diag.get("resource_preservation_required") is True
        and constrained_diag.get("constrained_solver_optimality_certified") is True
        and int(constrained_diag.get("max_dl_repair_steps", -1)) == 0
        and int(constrained_diag.get("max_dl_feasible_ascent_steps", -1)) == 0
        and constrained_result["selected_count"] == 2
        and constrained_result["reserved_cost_milli"] >= constrained_baseline["reserved_cost_milli"]
        and brute_best is not None
        and [row["ticker"] for row in constrained_result["selected_rows"]]
        == [row["ticker"] for row in brute_best["selected_rows"]],
    )

    score_constrained_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-score-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in constrained_seed
    ]
    score_constrained_order, score_constrained_diag = reorder_candidates_for_resource_aware_quality(
        score_constrained_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    score_constrained_action = select_resource_aware_action_candidates(
        score_constrained_order, score_constrained_diag
    )
    score_constrained_result = _simulate_reserved_candidate_order(
        score_constrained_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    score_constrained_baseline = _simulate_reserved_candidate_order(
        score_constrained_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    score_base_rank = {id(row): idx for idx, row in enumerate(score_constrained_rows)}
    score_brute_best = None
    score_brute_best_key = None
    for combo in itertools.combinations(score_constrained_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=score_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if (
            combo_result["selected_count"] != 2
            or combo_result["reserved_cost_milli"] < score_constrained_baseline["reserved_cost_milli"]
        ):
            continue
        combo_key = _max_dl_basket_quality_key(
            combo_result["selected_rows"], base_rank=score_base_rank
        )
        if score_brute_best_key is None or combo_key > score_brute_best_key:
            score_brute_best_key = combo_key
            score_brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_score_constrained_solver_matches_exhaustive_canonical_k_r0_oracle_and_reuses_exact_solver_without_calibration",
        True,
        score_constrained_diag.get("basket_objective") == "score"
        and score_constrained_diag.get("resource_preservation_required") is True
        and score_constrained_diag.get("constrained_solver_optimality_certified") is True
        and score_constrained_diag.get("constrained_solver_seed_source") == "c35-feasible-ascent"
        and int(score_constrained_diag.get("max_dl_repair_steps", -1)) == 0
        and int(score_constrained_diag.get("max_dl_feasible_ascent_steps", -1)) == 0
        and score_constrained_result["selected_count"] == 2
        and score_constrained_result["reserved_cost_milli"] >= score_constrained_baseline["reserved_cost_milli"]
        and score_brute_best is not None
        and [row["ticker"] for row in score_constrained_result["selected_rows"]]
        == [row["ticker"] for row in score_brute_best["selected_rows"]],
    )

    dual_safety_policy = "resource-aware-continuous-score-safety-constrained-optimal"
    dual_safety_seed = (
        ("B1", 100.0, 1200, 0.10, 0.80),
        ("B2", 100.0, 1100, 0.20, 0.70),
        ("K1", 100.0, 1300, 0.99, 0.20),
        ("K2", 100.0, 1200, 0.98, 0.30),
        ("S1", 100.0, 1300, 0.90, 0.90),
        ("S2", 100.0, 1200, 0.85, 0.80),
    )
    dual_safety_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score, dual_safety_policy, safety_score=safety_score
        )
        for ticker, price, qty, score, safety_score in dual_safety_seed
    ]
    dual_safety_baseline = _simulate_reserved_candidate_order(
        dual_safety_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    dual_safety_order, dual_safety_diag = reorder_candidates_for_resource_aware_quality(
        dual_safety_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    dual_safety_action = select_resource_aware_action_candidates(
        dual_safety_order, dual_safety_diag
    )
    dual_safety_result = _simulate_reserved_candidate_order(
        dual_safety_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    dual_base_rank = {id(row): idx for idx, row in enumerate(dual_safety_rows)}
    dual_floor_count = sum(
        int(row.get("breakout_quality_safety_score_available", False))
        for row in dual_safety_baseline["selected_rows"]
    )
    dual_floor_sum = sum(
        float(row.get("breakout_quality_safety_score") or 0.0)
        for row in dual_safety_baseline["selected_rows"]
        if row.get("breakout_quality_safety_score_available", False)
    )
    dual_brute_best = None
    dual_brute_key = None
    for combo in itertools.combinations(dual_safety_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=dual_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if (
            combo_result["selected_count"] != 2
            or combo_result["reserved_cost_milli"] < dual_safety_baseline["reserved_cost_milli"]
        ):
            continue
        selected = list(combo_result["selected_rows"] or [])
        safety_count = sum(
            int(row.get("breakout_quality_safety_score_available", False))
            for row in selected
        )
        safety_sum = sum(
            float(row.get("breakout_quality_safety_score") or 0.0)
            for row in selected
            if row.get("breakout_quality_safety_score_available", False)
        )
        if safety_count < dual_floor_count or safety_sum + 1e-12 < dual_floor_sum:
            continue
        combo_key = _max_dl_basket_quality_key(selected, base_rank=dual_base_rank)
        if dual_brute_key is None or combo_key > dual_brute_key:
            dual_brute_key = combo_key
            dual_brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "dual_model_safety_exact_solver_preserves_baseline_safety_floor_and_maximizes_primary_score_without_weighting",
        True,
        dual_safety_diag.get("basket_objective") == "score"
        and dual_safety_diag.get("safety_floor_enabled") is True
        and dual_safety_diag.get("safety_floor_contract") == "baseline_coverage_and_score_sum_floor_v1"
        and dual_safety_diag.get("safety_floor_violation") is False
        and dual_safety_diag.get("constrained_solver_optimality_certified") is True
        and dual_safety_diag.get("selected_safety_scored_count") >= dual_safety_diag.get("baseline_safety_scored_count")
        and float(dual_safety_diag.get("selected_safety_score_sum")) + 1e-12 >= float(dual_safety_diag.get("baseline_safety_score_sum"))
        and dual_brute_best is not None
        and [row["ticker"] for row in dual_safety_result["selected_rows"]]
        == [row["ticker"] for row in dual_brute_best["selected_rows"]]
        == ["S1", "S2"],
    )

    residual_safety_policy = "resource-aware-continuous-score-residual-safety-constrained-optimal"
    residual_seed = (
        ("B1", 100.0, 1200, 0.10, 0.60),
        ("B2", 100.0, 1100, 0.20, 0.55),
        ("K1", 100.0, 1300, 0.99, 0.20),
        ("K2", 100.0, 1200, 0.98, 0.25),
        ("S1", 100.0, 1300, 0.90, 0.60),
        ("S2", 100.0, 1200, 0.85, 0.55),
    )
    residual_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score, residual_safety_policy, safety_score=safety_score
        )
        for ticker, price, qty, score, safety_score in residual_seed
    ]
    residual_oracle_rows, residual_fit = _decorate_same_day_rank_residual_safety_scores(
        residual_rows
    )
    residual_baseline = _simulate_reserved_candidate_order(
        residual_oracle_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    residual_order, residual_diag = reorder_candidates_for_resource_aware_quality(
        residual_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    residual_action = select_resource_aware_action_candidates(
        residual_order, residual_diag
    )
    residual_result = _simulate_reserved_candidate_order(
        residual_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    residual_base_rank = {id(row): idx for idx, row in enumerate(residual_oracle_rows)}
    residual_floor_rows = list(residual_baseline["selected_rows"] or [])
    residual_floor_count = sum(
        int(row.get("breakout_quality_residual_safety_score_available", False))
        for row in residual_floor_rows
    )
    residual_floor_sum = sum(
        float(row.get("breakout_quality_residual_safety_score") or 0.0)
        for row in residual_floor_rows
        if row.get("breakout_quality_residual_safety_score_available", False)
    )
    residual_brute_best = None
    residual_brute_key = None
    for combo in itertools.combinations(residual_oracle_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=residual_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if (
            combo_result["selected_count"] != 2
            or combo_result["reserved_cost_milli"] < residual_baseline["reserved_cost_milli"]
        ):
            continue
        selected = list(combo_result["selected_rows"] or [])
        residual_count = sum(
            int(row.get("breakout_quality_residual_safety_score_available", False))
            for row in selected
        )
        residual_sum = sum(
            float(row.get("breakout_quality_residual_safety_score") or 0.0)
            for row in selected
            if row.get("breakout_quality_residual_safety_score_available", False)
        )
        if residual_count < residual_floor_count or residual_sum + 1e-12 < residual_floor_sum:
            continue
        combo_key = _max_dl_basket_quality_key(selected, base_rank=residual_base_rank)
        if residual_brute_key is None or combo_key > residual_brute_key:
            residual_brute_key = combo_key
            residual_brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "dual_model_residual_safety_uses_same_day_rank_ols_floor_and_preserves_primary_exact_objective",
        True,
        residual_diag.get("basket_objective") == "score"
        and residual_diag.get("safety_score_mode") == "residual"
        and residual_diag.get("safety_floor_contract") == "baseline_residual_coverage_and_score_sum_floor_v1"
        and residual_diag.get("residual_safety_transform") == "same_day_rank_ols_v1"
        and int(residual_diag.get("residual_safety_fit_pair_count", 0)) == 6
        and math.isclose(
            float(residual_diag.get("residual_safety_fit_slope")),
            float(residual_fit.get("residual_safety_fit_slope")),
            rel_tol=0.0, abs_tol=1e-12,
        )
        and residual_diag.get("safety_floor_violation") is False
        and residual_diag.get("constrained_solver_optimality_certified") is True
        and residual_brute_best is not None
        and [row["ticker"] for row in residual_result["selected_rows"]]
        == [row["ticker"] for row in residual_brute_best["selected_rows"]]
        and "K1" in [row["ticker"] for row in residual_result["selected_rows"]]
        and "S1" in [row["ticker"] for row in residual_result["selected_rows"]],
    )

    score_no_r0_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-score-no-r0-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in constrained_seed
    ]
    score_no_r0_order, score_no_r0_diag = reorder_candidates_for_resource_aware_quality(
        score_no_r0_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    score_no_r0_action = select_resource_aware_action_candidates(
        score_no_r0_order, score_no_r0_diag
    )
    score_no_r0_result = _simulate_reserved_candidate_order(
        score_no_r0_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    score_no_r0_base_rank = {id(row): idx for idx, row in enumerate(score_no_r0_rows)}
    score_no_r0_brute_best = None
    score_no_r0_brute_key = None
    for combo in itertools.combinations(score_no_r0_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=score_no_r0_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if combo_result["selected_count"] != 2:
            continue
        combo_key = _max_dl_basket_quality_key(
            combo_result["selected_rows"], base_rank=score_no_r0_base_rank
        )
        if score_no_r0_brute_key is None or combo_key > score_no_r0_brute_key:
            score_no_r0_brute_key = combo_key
            score_no_r0_brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_score_no_r0_exact_matches_exhaustive_k_cash_oracle_without_r0_repair",
        True,
        score_no_r0_diag.get("basket_objective") == "score"
        and score_no_r0_diag.get("resource_preservation_required") is False
        and score_no_r0_diag.get("constrained_solver_optimality_certified") is True
        and score_no_r0_diag.get("constrained_solver_seed_source")
        == "objective-top-k-if-cash-feasible-else-baseline"
        and int(score_no_r0_diag.get("max_dl_repair_steps", -1)) == 0
        and int(score_no_r0_diag.get("max_dl_feasible_ascent_steps", -1)) == 0
        and score_no_r0_result["selected_count"] == 2
        and score_no_r0_brute_best is not None
        and [row["ticker"] for row in score_no_r0_result["selected_rows"]]
        == [row["ticker"] for row in score_no_r0_brute_best["selected_rows"]],
    )

    score_capital_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-score-capital-no-r0-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in constrained_seed
    ]
    score_capital_order, score_capital_diag = reorder_candidates_for_resource_aware_quality(
        score_capital_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    score_capital_action = select_resource_aware_action_candidates(
        score_capital_order, score_capital_diag
    )
    score_capital_result = _simulate_reserved_candidate_order(
        score_capital_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    score_capital_base_rank = {id(row): idx for idx, row in enumerate(score_capital_rows)}
    score_capital_brute_best = None
    score_capital_brute_key = None
    for combo in itertools.combinations(score_capital_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=score_capital_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if combo_result["selected_count"] != 2:
            continue
        combo_key = _score_capital_basket_quality_key(
            combo_result, base_rank=score_capital_base_rank
        )
        if score_capital_brute_key is None or combo_key > score_capital_brute_key:
            score_capital_brute_key = combo_key
            score_capital_brute_best = combo_result
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_score_times_canonical_reserved_capital_no_r0_exact_matches_exhaustive_k_cash_oracle",
        True,
        score_capital_diag.get("basket_objective") == "score_capital"
        and score_capital_diag.get("resource_preservation_required") is False
        and score_capital_diag.get("constrained_solver_optimality_certified") is True
        and int(score_capital_diag.get("max_dl_repair_steps", -1)) == 0
        and int(score_capital_diag.get("max_dl_feasible_ascent_steps", -1)) == 0
        and score_capital_result["selected_count"] == 2
        and score_capital_brute_best is not None
        and [row["ticker"] for row in score_capital_result["selected_rows"]]
        == [row["ticker"] for row in score_capital_brute_best["selected_rows"]],
    )

    pareto_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-score-capital-pareto-no-r0-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in constrained_seed
    ]
    pareto_order, pareto_diag = reorder_candidates_for_resource_aware_quality(
        pareto_rows,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        pre_market_occupied=8,
        max_positions=10,
        params=resource_params,
    )
    pareto_action = select_resource_aware_action_candidates(pareto_order, pareto_diag)
    pareto_result = _simulate_reserved_candidate_order(
        pareto_action,
        available_cash=350_000.0,
        sizing_equity=2_000_000.0,
        free_slots=2,
        params=resource_params,
    )
    pareto_base_rank = {id(row): idx for idx, row in enumerate(pareto_rows)}

    def _pareto_oracle_summary(result):
        selected_rows = list(result.get("selected_rows") or [])
        scores = [
            float(row["breakout_quality_score"])
            for row in selected_rows
            if bool((row.get("breakout_quality_rank") or {}).get("available", False))
        ]
        stable = tuple(
            -pareto_base_rank[id(row)]
            for row in sorted(selected_rows, key=lambda item: pareto_base_rank[id(item)])
        )
        return (
            int(len(scores)),
            float(sum(scores)),
            int(result.get("reserved_cost_milli", 0) or 0),
            stable,
        )

    def _pareto_oracle_norm(value, low, high):
        if float(high) <= float(low) + 1e-12:
            return 1.0
        return min(1.0, max(0.0, (float(value) - float(low)) / (float(high) - float(low))))

    pareto_feasible = []
    for combo in itertools.combinations(pareto_rows, 2):
        combo_order = _max_dl_execution_order(combo, base_rank=pareto_base_rank)
        combo_result = _simulate_reserved_candidate_order(
            combo_order,
            available_cash=350_000.0,
            sizing_equity=2_000_000.0,
            free_slots=2,
            params=resource_params,
        )
        if combo_result["selected_count"] == 2:
            pareto_feasible.append(combo_result)
    pareto_score_endpoint = max(
        pareto_feasible,
        key=lambda result: (
            _pareto_oracle_summary(result)[0],
            _pareto_oracle_summary(result)[1],
            _pareto_oracle_summary(result)[2],
            _pareto_oracle_summary(result)[3],
        ),
    )
    pareto_capital_endpoint = max(
        pareto_feasible,
        key=lambda result: (
            _pareto_oracle_summary(result)[0],
            _pareto_oracle_summary(result)[2],
            _pareto_oracle_summary(result)[1],
            _pareto_oracle_summary(result)[3],
        ),
    )
    _score_cov, pareto_quality_max, pareto_capital_min, _ = _pareto_oracle_summary(pareto_score_endpoint)
    _capital_cov, pareto_quality_min, pareto_capital_max, _ = _pareto_oracle_summary(pareto_capital_endpoint)
    def _pareto_oracle_final_key(result):
        coverage, quality, capital_milli, stable = _pareto_oracle_summary(result)
        quality_norm = _pareto_oracle_norm(quality, pareto_quality_min, pareto_quality_max)
        capital_norm = _pareto_oracle_norm(capital_milli, pareto_capital_min, pareto_capital_max)
        return (coverage, quality_norm * capital_norm, quality, capital_milli, stable)

    pareto_brute_best = max(pareto_feasible, key=_pareto_oracle_final_key)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "mr13e_basket_level_pareto_no_r0_exact_matches_exhaustive_normalized_quality_capital_oracle",
        True,
        pareto_diag.get("basket_objective") == "score_capital_pareto"
        and pareto_diag.get("pareto_selection_method") == "normalized_product_v1"
        and pareto_diag.get("resource_preservation_required") is False
        and pareto_diag.get("constrained_solver_optimality_certified") is True
        and int(pareto_diag.get("max_dl_repair_steps", -1)) == 0
        and int(pareto_diag.get("max_dl_feasible_ascent_steps", -1)) == 0
        and pareto_result["selected_count"] == 2
        and [row["ticker"] for row in pareto_result["selected_rows"]]
        == [row["ticker"] for row in pareto_brute_best["selected_rows"]]
        and math.isclose(float(pareto_diag.get("pareto_quality_min")), float(pareto_quality_min))
        and math.isclose(float(pareto_diag.get("pareto_quality_max")), float(pareto_quality_max))
        and int(pareto_diag.get("pareto_capital_min_milli")) == int(pareto_capital_min)
        and int(pareto_diag.get("pareto_capital_max_milli")) == int(pareto_capital_max),
    )

    multi_swap_seed = (
        ("MS0", 100.0, 771, 1.0, -0.2293296791117404),
        ("MS1", 30.0, 1447, 0.9166666666666666, -0.10787340892209005),
        ("MS2", 100.0, 1243, 0.8333333333333334, -0.18915965847725308),
        ("MS3", 200.0, 384, 0.75, -0.04889139026396111),
        ("MS4", 50.0, 311, 0.6666666666666667, 0.3911970257315725),
        ("MS5", 30.0, 1671, 0.5833333333333333, 0.17800552639028372),
        ("MS6", 30.0, 1475, 0.5, 0.29724525296689247),
        ("MS7", 100.0, 1775, 0.41666666666666663, -0.12516494788049282),
        ("MS8", 50.0, 1811, 0.33333333333333337, 0.7644628747665172),
        ("MS9", 150.0, 700, 0.25, 0.6021816948230445),
    )
    c39_multi_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-feasible-ascent",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in multi_swap_seed
    ]
    c41_multi_rows = [
        _resource_candidate_fixed(
            ticker, price, qty, score,
            "resource-aware-continuous-excess-alpha-constrained-optimal",
            expected_excess_r=expected_excess_r,
        )
        for ticker, price, qty, score, expected_excess_r in multi_swap_seed
    ]
    c39_multi_order, c39_multi_diag = reorder_candidates_for_resource_aware_quality(
        c39_multi_rows, available_cash=350_000.0, sizing_equity=2_000_000.0,
        pre_market_occupied=6, max_positions=10, params=resource_params,
    )
    c41_multi_order, c41_multi_diag = reorder_candidates_for_resource_aware_quality(
        c41_multi_rows, available_cash=350_000.0, sizing_equity=2_000_000.0,
        pre_market_occupied=6, max_positions=10, params=resource_params,
    )
    c39_multi_action = select_resource_aware_action_candidates(
        c39_multi_order, c39_multi_diag
    )
    c41_multi_action = select_resource_aware_action_candidates(
        c41_multi_order, c41_multi_diag
    )
    c39_multi_result = _simulate_reserved_candidate_order(
        c39_multi_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=4, params=resource_params,
    )
    c41_multi_result = _simulate_reserved_candidate_order(
        c41_multi_action, available_cash=350_000.0, sizing_equity=2_000_000.0,
        free_slots=4, params=resource_params,
    )
    c39_multi_rank = {id(row): idx for idx, row in enumerate(c39_multi_rows)}
    c41_multi_rank = {id(row): idx for idx, row in enumerate(c41_multi_rows)}
    c39_multi_key = _excess_alpha_basket_quality_key(
        c39_multi_result, base_rank=c39_multi_rank
    )
    c41_multi_key = _excess_alpha_basket_quality_key(
        c41_multi_result, base_rank=c41_multi_rank
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "excess_alpha_exact_solver_escapes_c39_one_swap_local_optimum_with_strictly_better_multi_swap_feasible_basket",
        True,
        [row["ticker"] for row in c39_multi_result["selected_rows"]]
        == ["MS2", "MS5", "MS8", "MS9"]
        and [row["ticker"] for row in c41_multi_result["selected_rows"]]
        == ["MS0", "MS3", "MS8", "MS9"]
        and len(
            set(row["ticker"] for row in c39_multi_result["selected_rows"])
            ^ set(row["ticker"] for row in c41_multi_result["selected_rows"])
        ) == 4
        and c41_multi_key > c39_multi_key
        and c41_multi_diag.get("constrained_solver_optimality_certified") is True
        and int(c41_multi_diag.get("max_dl_repair_steps", -1)) == 0
        and int(c41_multi_diag.get("max_dl_feasible_ascent_steps", -1)) == 0,
    )

    selection_excess_settings = strategy_config.get_strategy_comparison_settings("selection_pit")
    c28 = selection_excess_settings.arms["C28"]
    c39 = selection_excess_settings.arms["C39"]
    c40 = selection_excess_settings.arms["C40"]
    c41 = selection_excess_settings.arms["C41"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c39_is_selection_only_frozen_mr13e_excess_alpha_with_same_k_r0_and_direct_c35_contrast",
        True,
        not c39.enabled
        and c39.dl_id == "CONT13E_PIT"
        and c39.dl_runtime_mode == "resource-aware-continuous-excess-alpha-feasible-ascent"
        and dict(c39.dl_runtime_options or {}).get("expected_excess_r_fit_dl_id") == "CONT13E_PIT"
        and dict(c39.dl_runtime_options or {}).get("preserve_k_r0") is True
        and dict(c39.dl_runtime_options or {}).get("negative_expected_excess_r_allowed") is True
        and dict(c39.dl_runtime_options or {}).get("selection_only") is True
        and c39.robustness_role == "off"
        and "C39-C35" in selection_excess_settings.contrasts
        and "C39" not in {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms},
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c40_is_selection_only_c39_no_r0_ablation_with_same_expected_excess_alpha_objective_and_k",
        True,
        not c40.enabled
        and c40.dl_id == c39.dl_id == "CONT13E_PIT"
        and c40.dl_runtime_mode == "resource-aware-continuous-excess-alpha-no-r0-feasible-ascent"
        and dict(c40.dl_runtime_options or {}).get("expected_excess_r_fit_dl_id") == "CONT13E_PIT"
        and dict(c40.dl_runtime_options or {}).get("preserve_k") is True
        and dict(c40.dl_runtime_options or {}).get("preserve_r0") is False
        and dict(c40.dl_runtime_options or {}).get("r0_minimum_repair") is False
        and dict(c40.dl_runtime_options or {}).get("negative_expected_excess_r_allowed") is True
        and dict(c40.dl_runtime_options or {}).get("selection_only") is True
        and c40.robustness_role == "off"
        and "C40-C39" in selection_excess_settings.contrasts
        and "C40-C35" in selection_excess_settings.contrasts
        and "C40" not in {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms},
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c41_is_selection_only_exact_constrained_solver_ablation_with_same_c39_objective_k_r0_and_calibration",
        True,
        not c41.enabled
        and c41.dl_id == c39.dl_id == "CONT13E_PIT"
        and c41.dl_runtime_mode == "resource-aware-continuous-excess-alpha-constrained-optimal"
        and dict(c41.dl_runtime_options or {}).get("expected_excess_r_fit_dl_id") == "CONT13E_PIT"
        and dict(c41.dl_runtime_options or {}).get("preserve_k_r0") is True
        and dict(c41.dl_runtime_options or {}).get("constrained_solver") == "exact_branch_and_bound_v1"
        and dict(c41.dl_runtime_options or {}).get("negative_expected_excess_r_allowed") is True
        and dict(c41.dl_runtime_options or {}).get("selection_only") is True
        and c41.robustness_role == "off"
        and "C41-C39" in selection_excess_settings.contrasts
        and "C41-C35" in selection_excess_settings.contrasts
        and "C41" not in {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms},
    )

    c42 = selection_excess_settings.arms["C42"]
    c42_options = dict(c42.dl_runtime_options or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c42_is_selection_only_mr13e_score_exact_constrained_solver_ablation_without_expected_r_calibration",
        True,
        c42.enabled
        and c42.dl_id == c41.dl_id == c39.dl_id == "CONT13E_PIT"
        and c42.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and c42_options.get("preserve_k_r0") is True
        and c42_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c42_options.get("selection_only") is True
        and not any(
            key in c42_options
            for key in (
                "expected_excess_r_fit_dl_id",
                "expected_excess_r_calibration_method",
                "expected_r_fit_dl_id",
                "expected_r_calibration_method",
            )
        )
        and c42.robustness_role == "off"
        and "C42-C35" in selection_excess_settings.contrasts
        and "C42-C41" in selection_excess_settings.contrasts
        and {"C42-C23", "C42-C32"}.issubset(
            {contrast.contrast_id for contrast in selection_excess_settings.enabled_contrasts}
        )
        and "C42" not in {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms},
    )

    c46 = selection_excess_settings.arms["C46"]
    c47 = selection_excess_settings.arms["C47"]
    c48 = selection_excess_settings.arms["C48"]
    c48_options = dict(c48.dl_runtime_options or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c48_rejected_selection_only_pareto_no_r0_remains_historical_and_current_selection_stays_minimal",
        True,
        not c46.enabled
        and not c47.enabled
        and not c48.enabled
        and c48.dl_id == c42.dl_id == "CONT13E_PIT"
        and c48.dl_runtime_mode == "resource-aware-continuous-score-capital-pareto-no-r0-constrained-optimal"
        and c48_options.get("preserve_k") is True
        and c48_options.get("preserve_r0") is False
        and c48_options.get("r0_minimum_repair") is False
        and c48_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c48_options.get("pareto_selection") == "normalized_product_v1"
        and c48_options.get("pareto_quality") == "score_sum_max_coverage_first"
        and c48_options.get("pareto_capital") == "canonical_reserved_cost_milli"
        and c48_options.get("selection_only") is True
        and {"C32", "C23", "C42", "C57"}
        == {arm.arm_id for arm in selection_excess_settings.enabled_arms}
        and "C48-C42" not in {
            contrast.contrast_id for contrast in selection_excess_settings.enabled_contrasts
        }
        and "C48-C42" in selection_excess_settings.contrasts
        and "C48" not in {
            arm.arm_id
            for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms
        },
    )

    c53 = selection_excess_settings.arms["C53"]
    c53_options = dict(c53.dl_runtime_options or {})
    c57 = selection_excess_settings.arms["C57"]
    c57_options = dict(c57.dl_runtime_options or {})
    c51_historical = selection_excess_settings.arms["C51"]
    c52_historical = selection_excess_settings.arms["C52"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c57_is_current_c56_selection_counterpart_while_c53_source_only_arm_remains_historical",
        True,
        not c53.enabled
        and c53.param_source == c42.param_source == "selection_min_roos"
        and c53.rule_policy == c42.rule_policy == "all_off"
        and c53.dl_id == "CONT13K_PIT"
        and c53.dl_runtime_mode == c42.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and c53_options == c42_options
        and c57.enabled
        and c57.param_source == c42.param_source == "selection_min_roos"
        and c57.rule_policy == c42.rule_policy == "all_off"
        and c57.dl_id == "CONT13K_PIT"
        and c57.dl_runtime_mode == "resource-aware-continuous-score-residual-safety-constrained-optimal"
        and c57_options.get("safety_dl_id") == "CONT13M_PIT"
        and c57_options.get("safety_constraint") == "baseline_residual_coverage_and_score_sum_floor_v1"
        and c57_options.get("safety_residualization") == "same_day_rank_ols_v1"
        and c57_options.get("preserve_k_r0") is True
        and c57_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c57_options.get("selection_only") is True
        and {"C57-C42", "C57-C23", "C57-C32"}.issubset(
            {contrast.contrast_id for contrast in selection_excess_settings.enabled_contrasts}
        )
        and {"C32", "C23", "C42", "C57"}
        == {arm.arm_id for arm in selection_excess_settings.enabled_arms}
        and not c51_historical.enabled
        and not c52_historical.enabled
        and "CONT13H_PIT" in strategy_config.HISTORICAL_STRATEGY_DL_SOURCES
        and "CONT13H" in strategy_config.HISTORICAL_STRATEGY_DL_SOURCES
        and "C51" in strategy_config.HISTORICAL_STRATEGY_COMPARE_ARMS
        and "C52" in strategy_config.HISTORICAL_STRATEGY_COMPARE_ARMS,
    )

    forward_current_settings = strategy_config.get_strategy_comparison_settings("forward_oos")
    c44_current = forward_current_settings.arms["C44"]
    c54_current = forward_current_settings.arms["C54"]
    c55_current = forward_current_settings.arms["C55"]
    c56_current = forward_current_settings.arms["C56"]
    c44_options = dict(c44_current.dl_runtime_options or {})
    c54_options = dict(c54_current.dl_runtime_options or {})
    c55_options = dict(c55_current.dl_runtime_options or {})
    c56_options = dict(c56_current.dl_runtime_options or {})
    operational_current_settings = strategy_config.get_strategy_comparison_settings("extending_window_rolling")
    c61_current = operational_current_settings.arms["C61"]
    c58_current = operational_current_settings.arms["C58"]
    c59_current = operational_current_settings.arms["C59"]
    c60_current = operational_current_settings.arms["C60"]
    c60_options = dict(c60_current.dl_runtime_options or {})
    operational_robustness_active = strategy_config.get_strategy_multi_seed_robustness_settings("extending_window_rolling")
    extending_param_source = operational_current_settings.parameter_sources["extending_min_roos"]
    extending_param_builder = extending_param_source.builder
    extending_full_source = operational_current_settings.parameter_sources["extending_full_roos"]
    extending_full_builder = extending_full_source.builder
    selection_robustness_legacy = strategy_config.get_strategy_multi_seed_robustness_settings("selection_pit")
    forward_robustness_legacy = strategy_config.get_strategy_multi_seed_robustness_settings("forward_oos")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "operational_c60_is_current_dual_model_research_line_while_c56_c57_full_flow_is_legacy",
        True,
        {"C61", "C58", "C59", "C60"}
        == {arm.arm_id for arm in operational_current_settings.enabled_arms}
        and c61_current.enabled and not c61_current.dl_enabled
        and c61_current.param_source == "extending_full_roos"
        and c61_current.rule_policy == "formal"
        and c58_current.enabled and not c58_current.dl_enabled
        and c59_current.enabled and c59_current.dl_id == "CONT13E_ROLL"
        and c60_current.enabled
        and c60_current.param_source == c58_current.param_source == c59_current.param_source == "extending_min_roos"
        and extending_param_builder is not None
        and extending_param_builder.builder_type == "extending_min_roos_stitch"
        and dict(extending_param_builder.options).get("output_relative_dir")
            == "models/research/breakout_quality/strategy_compare/extending_min_roos"
        and extending_param_source.identity_manifest_path.endswith("extending_stitch_manifest.json")
        and dict(extending_param_source.artifact_contract.get("breakout_quality_param_adaptation") or {}).get("parameter_set")
            == "P2_EXTENDING"
        and extending_full_builder is not None
        and extending_full_builder.builder_type == "extending_full_roos_stitch"
        and dict(extending_full_builder.options).get("output_relative_dir")
            == "models/research/breakout_quality/strategy_compare/extending_full_roos"
        and extending_full_source.identity_manifest_path.endswith("extending_stitch_manifest.json")
        and dict(extending_full_source.artifact_contract.get("breakout_quality_param_adaptation") or {}).get("parameter_set")
            == "P4_EXTENDING"
        and c60_current.rule_policy == c58_current.rule_policy == c59_current.rule_policy == "all_off"
        and c60_current.dl_id == "CONT13K_ROLL"
        and c60_current.dl_runtime_mode == "resource-aware-continuous-score-residual-safety-constrained-optimal"
        and c60_options.get("safety_dl_id") == "CONT13M_ROLL"
        and c60_options.get("safety_constraint") == "baseline_residual_coverage_and_score_sum_floor_v1"
        and c60_options.get("safety_residualization") == "same_day_rank_ols_v1"
        and c60_options.get("preserve_k_r0") is True
        and c60_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c60_options.get("selection_only") is True
        and {"C61-C58", "C59-C58", "C59-C61", "C60-C58", "C60-C61", "C60-C59"}
        == {contrast.contrast_id for contrast in operational_current_settings.enabled_contrasts}
        and operational_robustness_active.enabled
        and tuple(operational_robustness_active.fixed_arm_ids) == ("C58",)
        and tuple(operational_robustness_active.stochastic_arm_ids) == ("C60",)
        and tuple(operational_robustness_active.paired_contrasts) == ()
        and not selection_robustness_legacy.enabled
        and not forward_robustness_legacy.enabled
        and c56_current.enabled
        and c44_current.enabled
        and not c54_current.enabled
        and not c55_current.enabled
        and operational_robustness_active.seed_count
        == strategy_config.STRATEGY_COMPARE_ROBUSTNESS_SEED_COUNT == 4
        and operational_robustness_active.seed_generator_seed
        == strategy_config.STRATEGY_COMPARE_ROBUSTNESS_SEED_GENERATOR_SEED == 20260810
        and strategy_config.get_strategy_runtime_integration_settings().selection_candidate_arm_id == "C42"
        and strategy_config.get_strategy_runtime_integration_settings().forward_candidate_arm_id == "C44",
    )

    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_required_artifact_sources,
    )
    from filters.breakout_quality.strategy_comparison import _resolved_ranking_options

    _, forward_required_dl, forward_runtime_dl = resolve_required_artifact_sources(
        forward_current_settings
    )
    _, selection_required_dl, selection_runtime_dl = resolve_required_artifact_sources(
        selection_excess_settings
    )
    c56_pinned_options = _resolved_ranking_options(
        forward_current_settings,
        c56_current,
        continuous_score_overrides={
            "CONT13M": {
                "score_path": "models/pinned_cont13m_forward.csv.gz",
                "manifest_path": "models/pinned_cont13m_forward_manifest.json",
            }
        },
    )
    c57_pinned_options = _resolved_ranking_options(
        selection_excess_settings,
        c57,
        continuous_score_overrides={
            "CONT13M_PIT": {
                "score_path": "models/pinned_cont13m_pit.csv.gz",
                "manifest_path": "models/pinned_cont13m_pit_manifest.json",
            }
        },
    )
    c55_reuse_source = reuse_path.read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "c56_c57_secondary_dl_sources_are_required_pinned_and_stage_matched",
        True,
        {"CONT13E", "CONT13K", "CONT13M"}.issubset(forward_required_dl)
        and {"CONT13E", "CONT13K", "CONT13M"}.issubset(forward_runtime_dl)
        and {"CONT13E_PIT", "CONT13K_PIT", "CONT13M_PIT"}.issubset(selection_required_dl)
        and {"CONT13E_PIT", "CONT13K_PIT", "CONT13M_PIT"}.issubset(selection_runtime_dl)
        and c56_pinned_options.get("safety_filter_id") == "breakout_quality_v1"
        and c56_pinned_options.get("safety_score_source") == "continuous_ranker_oos"
        and c56_pinned_options.get("safety_model_architecture") == "inception_time_v1"
        and c56_pinned_options.get("safety_experiment_profile")
        == "daily_universal_full_horizon_low_adverse_full_list_ndcg_pairwise"
        and c56_pinned_options.get("safety_score_path_override")
        == "models/pinned_cont13m_forward.csv.gz"
        and c56_pinned_options.get("safety_score_manifest_path_override")
        == "models/pinned_cont13m_forward_manifest.json"
        and c56_pinned_options.get("safety_residualization") == "same_day_rank_ols_v1"
        and c57_pinned_options.get("safety_filter_id") == "breakout_quality_v1"
        and c57_pinned_options.get("safety_score_source") == "selection_point_in_time"
        and c57_pinned_options.get("safety_score_path_override")
        == "models/pinned_cont13m_pit.csv.gz"
        and c57_pinned_options.get("safety_score_manifest_path_override")
        == "models/pinned_cont13m_pit_manifest.json"
        and c57_pinned_options.get("safety_residualization") == "same_day_rank_ols_v1"
        and '"safety_dl_source"' in c55_reuse_source
        and "safety_artifact_names" in c55_reuse_source,
    )


    from filters.breakout_quality import runtime as breakout_runtime
    c55_runtime_lookup_calls = []

    def _fake_c55_lookup(**kwargs):
        c55_runtime_lookup_calls.append(dict(kwargs))
        return {
            "available": True,
            "score": 0.75 if len(c55_runtime_lookup_calls) == 1 else 0.25,
            "score_date": str(kwargs.get("signal_date")),
            "score_source": "continuous_ranker_oos",
            "unavailable_reason": "",
        }

    with patch.object(
        breakout_runtime,
        "lookup_continuous_ranker_oos_candidate_score",
        side_effect=_fake_c55_lookup,
    ):
        with breakout_runtime.breakout_quality_ranking_source_context(
            score_source="continuous_ranker_oos",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
            ranking_policy="resource-aware-continuous-score-residual-safety-constrained-optimal",
            ranking_options=c56_pinned_options,
            score_path_override="models/pinned_cont13k_forward.csv.gz",
        ):
            c55_runtime_payload = breakout_runtime.resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date="2025-01-02",
                information_date="2025-01-02",
                high_len=60,
                project_root=str(project_root),
            )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "c56_runtime_looks_up_primary_and_secondary_frozen_scores_with_separate_pinned_paths",
        True,
        len(c55_runtime_lookup_calls) == 2
        and c55_runtime_lookup_calls[0].get("score_path_override")
        == "models/pinned_cont13k_forward.csv.gz"
        and c55_runtime_lookup_calls[1].get("score_path_override")
        == "models/pinned_cont13m_forward.csv.gz"
        and math.isclose(float(c55_runtime_payload.get("score")), 0.75)
        and math.isclose(float(c55_runtime_payload.get("safety_score")), 0.25)
        and c55_runtime_payload.get("safety_available") is True
        and c55_runtime_payload.get("safety_dl_id") == "CONT13M",
    )

    c57_runtime_lookup_calls = []

    def _fake_c57_pit_lookup(**kwargs):
        c57_runtime_lookup_calls.append(dict(kwargs))
        return {
            "available": True,
            "score": 0.80 if len(c57_runtime_lookup_calls) == 1 else 0.35,
            "score_date": str(kwargs.get("signal_date")),
            "score_source": "selection_point_in_time",
            "unavailable_reason": "",
        }

    with patch.object(
        breakout_runtime,
        "lookup_selection_point_in_time_candidate_score",
        side_effect=_fake_c57_pit_lookup,
    ):
        with breakout_runtime.breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_full_horizon_pure_mfe_full_list_ndcg_pairwise",
            ranking_policy="resource-aware-continuous-score-residual-safety-constrained-optimal",
            ranking_options=c57_pinned_options,
            score_path_override="models/pinned_cont13k_pit.csv.gz",
            score_manifest_path_override="models/pinned_cont13k_pit_manifest.json",
        ):
            c57_runtime_payload = breakout_runtime.resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date="2020-01-02",
                information_date="2020-01-02",
                high_len=60,
                project_root=str(project_root),
            )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "c57_runtime_looks_up_primary_and_secondary_pit_scores_with_separate_pinned_score_and_manifest_paths",
        True,
        len(c57_runtime_lookup_calls) == 2
        and c57_runtime_lookup_calls[0].get("score_path_override")
        == "models/pinned_cont13k_pit.csv.gz"
        and c57_runtime_lookup_calls[0].get("manifest_path_override")
        == "models/pinned_cont13k_pit_manifest.json"
        and c57_runtime_lookup_calls[1].get("score_path_override")
        == "models/pinned_cont13m_pit.csv.gz"
        and c57_runtime_lookup_calls[1].get("manifest_path_override")
        == "models/pinned_cont13m_pit_manifest.json"
        and c57_runtime_lookup_calls[1].get("filter_id") == "breakout_quality_v1"
        and math.isclose(float(c57_runtime_payload.get("score")), 0.80)
        and math.isclose(float(c57_runtime_payload.get("safety_score")), 0.35)
        and c57_runtime_payload.get("safety_available") is True
        and c57_runtime_payload.get("safety_dl_id") == "CONT13M_PIT",
    )


    from filters.breakout_quality.strategy_compare_diagnostics import (
        _continuous_forward_target_lookup,
        _strategy_selection_diagnostics,
        render_strategy_r_analysis_table,
    )

    with tempfile.TemporaryDirectory() as forward_diag_tmp:
        forward_diag_score_path = Path(forward_diag_tmp) / "daily_ranker_oos_scores.csv.gz"
        pd.DataFrame([
            {
                "ticker": "A", "date": "2021-01-04", "group_index": 1,
                "target_raw_r": 2.0, "target_daily_percentile": 1.0, "model_score": 0.9,
            },
            {
                "ticker": "B", "date": "2021-01-04", "group_index": 2,
                "target_raw_r": 0.0, "target_daily_percentile": 0.5, "model_score": 0.2,
            },
        ]).to_csv(
            forward_diag_score_path, index=False, encoding="utf-8-sig", compression="gzip"
        )
        forward_diag_lookup = _continuous_forward_target_lookup(
            root=project_root,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            architecture="inception_time_v1",
            profile=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            score_path_override=str(forward_diag_score_path),
        )
        forward_diag_orderable = pd.DataFrame([
            {
                "ticker": "A", "trade_date": "2021-01-04", "signal_date": "2021-01-04",
                "breakout_quality_score_date": "2021-01-04",
                "breakout_quality_score": 0.9, "breakout_quality_score_available": True,
            },
            {
                "ticker": "B", "trade_date": "2021-01-04", "signal_date": "2021-01-04",
                "breakout_quality_score_date": "2021-01-04",
                "breakout_quality_score": 0.2, "breakout_quality_score_available": True,
            },
        ])
        forward_diag_selected = pd.DataFrame([
            {"ticker": "A", "trade_date": "2021-01-04", "signal_date": "2021-01-04"},
        ])
        forward_diag_metrics, _, _ = _strategy_selection_diagnostics(
            orderable=forward_diag_orderable,
            selected=forward_diag_selected,
            lookup=forward_diag_lookup,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_strategy_report_joins_embedded_future_target_only_after_replay_for_selection_translation",
        True,
        math.isclose(float(forward_diag_metrics["orderable_score_coverage_rate"]), 1.0)
        and math.isclose(float(forward_diag_metrics["selected_target_mean_r"]), 2.0)
        and math.isclose(float(forward_diag_metrics["selected_target_percentile_mean"]), 1.0)
        and math.isclose(float(forward_diag_metrics["target_top_k_retention_mean"]), 1.0)
        and math.isclose(float(forward_diag_metrics["target_opportunity_gap_r_mean"]), 0.0)
        and bool(forward_diag_metrics["future_target_used_for_runtime_sort"]) is False
        and "score_source in {" in strategy_compare_source
        and "SCORE_SOURCE_CONTINUOUS_RANKER_OOS" in strategy_compare_source
        and "_continuous_forward_target_lookup(" in strategy_compare_source,
    )

    target_aware_report = render_strategy_r_analysis_table({
        "r_analysis": [
            {
                "arm_id": "A", "name": "Target-A", "continuous_target_id": "target_a",
                "portfolio_avg_r": 0.7, "mean_daily_spearman": 0.9,
                "selected_target_mean_r": 9.0,
            },
            {
                "arm_id": "B", "name": "Target-B1", "continuous_target_id": "target_b",
                "portfolio_avg_r": 0.7, "mean_daily_spearman": 0.1,
                "selected_target_mean_r": 1.0,
            },
            {
                "arm_id": "C", "name": "Target-B2", "continuous_target_id": "target_b",
                "portfolio_avg_r": 0.7, "mean_daily_spearman": 0.2,
                "selected_target_mean_r": 2.0,
            },
        ]
    }, target="markdown")
    target_aware_lines = {
        line.split()[0]: line
        for line in target_aware_report.splitlines()
        if line.startswith(("A ", "B ", "C "))
    }
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "r_analysis_best_worst_coloring_is_target_identity_aware_while_actual_trade_metrics_remain_cross_arm",
        True,
        "<span" not in target_aware_lines.get("A", "")
        and "<span" in target_aware_lines.get("B", "")
        and "<span" in target_aware_lines.get("C", ""),
    )


    c43 = selection_excess_settings.arms["C43"]
    c43_options = dict(c43.dl_runtime_options or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c43_is_selection_only_mr13a_score_exact_constrained_solver_ablation_using_same_generic_solver_as_c42",
        True,
        not c43.enabled
        and c43.dl_id == c28.dl_id == "CONT13A_PIT"
        and c43.param_source == c28.param_source == c42.param_source == "selection_min_roos"
        and c43.rule_policy == c28.rule_policy == c42.rule_policy == "all_off"
        and c43.dl_runtime_mode == c42.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and c43_options == c42_options
        and c43_options.get("preserve_k_r0") is True
        and c43_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c43_options.get("selection_only") is True
        and not any(
            key in c43_options
            for key in (
                "expected_excess_r_fit_dl_id",
                "expected_excess_r_calibration_method",
                "expected_r_fit_dl_id",
                "expected_r_calibration_method",
            )
        )
        and c43.robustness_role == "off"
        and "C43-C28" in selection_excess_settings.contrasts
        and "C42-C43" in selection_excess_settings.contrasts
        and "C43" not in {arm.arm_id for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms},
    )

    c32_full = selection_excess_settings.arms["C32"]
    c45 = selection_excess_settings.arms["C45"]
    c45_options = dict(c45.dl_runtime_options or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c45_rejected_full_roos_transfer_is_historical_only_and_preserves_exact_solver_identity",
        True,
        not c45.enabled
        and c45.dl_id == c42.dl_id == "CONT13E_PIT"
        and c45.param_source == c32_full.param_source == "selection_full_roos"
        and c45.rule_policy == c32_full.rule_policy == "formal"
        and c45.dl_runtime_mode == c42.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and c45_options == c42_options
        and c45_options.get("preserve_k_r0") is True
        and c45_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c45_options.get("selection_only") is True
        and c45.robustness_role == "off"
        and {"C45-C32", "C45-C42"}.issubset(selection_excess_settings.contrasts)
        and not {"C45-C32", "C45-C42"}.intersection(
            {contrast.contrast_id for contrast in selection_excess_settings.enabled_contrasts}
        )
        and "C45" not in {arm.arm_id for arm in selection_excess_settings.enabled_arms}
        and "C45" not in {
            arm.arm_id
            for arm in strategy_config.get_strategy_comparison_settings("forward_oos").enabled_arms
        },
    )

    forward_exact_settings = strategy_config.get_strategy_comparison_settings("forward_oos")
    c20_forward = forward_exact_settings.arms["C20"]
    c36_forward = forward_exact_settings.arms["C36"]
    c44 = forward_exact_settings.arms["C44"]
    c44_options = dict(c44.dl_runtime_options or {})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "sr_c44_is_forward_mr13e_score_exact_constrained_solver_ablation_using_same_generic_solver_as_c42_and_retains_mr12b_anchor",
        True,
        c44.enabled
        and c44.dl_id == c36_forward.dl_id == "CONT13E"
        and c44.param_source == c36_forward.param_source == c20_forward.param_source == "min_roos"
        and c44.rule_policy == c36_forward.rule_policy == c20_forward.rule_policy == "all_off"
        and c44.dl_runtime_mode == c42.dl_runtime_mode == "resource-aware-continuous-score-constrained-optimal"
        and c44_options.get("preserve_k_r0") is True
        and c44_options.get("constrained_solver") == "exact_branch_and_bound_v1"
        and c44_options.get("selection_only") is False
        and {k: v for k, v in c44_options.items() if k != "selection_only"}
            == {k: v for k, v in c42_options.items() if k != "selection_only"}
        and not any(
            key in c44_options
            for key in (
                "expected_excess_r_fit_dl_id",
                "expected_excess_r_calibration_method",
                "expected_r_fit_dl_id",
                "expected_r_calibration_method",
            )
        )
        and c44.robustness_role == "off"
        and not c20_forward.enabled
        and not c36_forward.enabled
        and c20_forward.dl_id == "CONT12B"
        and {"C44-C20", "C44-C36"}.issubset(forward_exact_settings.contrasts)
        and {"C44-C3", "C44-C1"}.issubset(
            {contrast.contrast_id for contrast in forward_exact_settings.enabled_contrasts}
        )
        and "C43" not in {arm.arm_id for arm in forward_exact_settings.enabled_arms},
    )

    from filters.breakout_quality.rank_calibration import (
        add_daily_excess_r,
        fit_weighted_increasing_isotonic,
    )
    canonical_excess = add_daily_excess_r(pd.DataFrame({
        "date": ["2020-01-02", "2020-01-02", "2020-01-02"],
        "target_raw_r": [0.0, 1.0, 5.0],
        "target_valid": [True, True, True],
    }))
    curve, curve_summary = fit_weighted_increasing_isotonic(
        np.asarray([0.2, 0.5, 0.8, 1.0], dtype=np.float64),
        np.asarray([-0.8, -0.1, 0.4, 0.2], dtype=np.float64),
    )
    curve_predictions = curve.predict(np.asarray([0.2, 0.5, 0.8, 1.0], dtype=np.float64))
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "excess_r_calibration_uses_full_daily_eligible_mean_and_monotonic_pava_without_absolute_r_baseline",
        True,
        math.isclose(float(canonical_excess["daily_target_mean_r"].iloc[0]), 2.0, abs_tol=1e-12)
        and np.allclose(
            canonical_excess["target_excess_r"].to_numpy(dtype=np.float64),
            np.asarray([-2.0, -1.0, 3.0], dtype=np.float64),
        )
        and math.isclose(float(canonical_excess["target_excess_r"].sum()), 0.0, abs_tol=1e-12)
        and bool((np.diff(curve_predictions) >= -1e-12).all())
        and int(curve_summary["isotonic_block_count"]) < 4,
    )

    selection_expected_settings = strategy_config.get_strategy_comparison_settings("selection_pit")
    forward_expected_settings = strategy_config.get_strategy_comparison_settings("forward_oos")
    c37 = selection_expected_settings.arms["C37"]
    c38 = forward_expected_settings.arms["C38"]
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "frozen_mr13e_expected_pnl_arms_keep_same_ranker_identity_and_pit_fit_source_contract",
        True,
        (not c37.enabled) and (not c38.enabled)
        and "C37" in strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS
        and "C38" in strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS
        and c37.dl_id == "CONT13E_PIT"
        and c38.dl_id == "CONT13E"
        and c37.dl_runtime_mode == c38.dl_runtime_mode
        == "resource-aware-continuous-expected-pnl-feasible-ascent"
        and dict(c37.dl_runtime_options or {}).get("expected_r_fit_dl_id") == "CONT13E_PIT"
        and dict(c38.dl_runtime_options or {}).get("expected_r_fit_dl_id") == "CONT13E_PIT"
        and dict(c37.dl_runtime_options or {}).get("preserve_k_r0") is True
        and dict(c38.dl_runtime_options or {}).get("preserve_k_r0") is True
        and c37.robustness_role == c38.robustness_role == "off",
    )

    from services.breakout_quality import expected_r_calibration as expected_r_calibration_service
    from filters.breakout_quality.expected_r_calibration import (
        validate_expected_r_calibration_artifact,
    )
    with tempfile.TemporaryDirectory() as calibration_tmp:
        calibration_root = Path(calibration_tmp)
        fit_score_path = calibration_root / "fit_scores.csv"
        runtime_score_path = calibration_root / "runtime_scores.csv"
        fit_score_path.write_text("fit", encoding="utf-8")
        runtime_score_path.write_text("runtime", encoding="utf-8")
        fit_scores = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
            "group_index": [1, 2, 3, 4],
            "breakout_quality_score": [0.1, 0.9, 0.2, 0.8],
        })
        runtime_scores = pd.DataFrame({
            "ticker": ["A", "B"],
            "date": ["2021-01-04", "2021-01-04"],
            "group_index": [101, 102],
            "breakout_quality_score": [0.3, 0.7],
        })
        fake_contract_fit = SimpleNamespace(score_path=fit_score_path)
        fake_contract_runtime = SimpleNamespace(
            score_path=runtime_score_path,
            execution_start="2021-01-01",
        )
        fake_bundle = SimpleNamespace(
            group_table=pd.DataFrame({
                "ticker": ["A", "B", "A", "B"],
                "date": ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"],
                "group_index": [1, 2, 3, 4],
                "label_eval_end_date": ["2020-02-15"] * 4,
            }),
            raw_target=np.asarray([0.0, 2.0, 0.2, 1.8], dtype=np.float64),
            target_valid=np.asarray([True, True, True, True], dtype=bool),
            target_manifest={"target_id": "daily_opportunity_no_time_r_v1"},
        )
        with patch.object(
            expected_r_calibration_service, "_selection_score_frame",
            return_value=(fit_scores, fake_contract_fit),
        ), patch.object(
            expected_r_calibration_service, "_runtime_score_frame",
            return_value=(runtime_scores, fake_contract_runtime, "continuous_ranker_oos"),
        ), patch.object(
            expected_r_calibration_service, "load_profile_continuous_ranker_data",
            return_value=fake_bundle,
        ):
            calibration_payload = expected_r_calibration_service.build_expected_r_calibration_artifact(
                project_root=calibration_root,
                filter_id="breakout_quality_v1",
                model_architecture="inception_time_v1",
                experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
                phase_id="forward_oos",
            )
        calibration_manifest = dict(calibration_payload["manifest"])
        calibration_lookup = pd.read_csv(calibration_payload["paths"]["lookup"])
        forward_cutoff_ready, _forward_cutoff_status, _ = validate_expected_r_calibration_artifact(
            calibration_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
            phase_id="forward_oos",
            expected_forward_frozen_cutoff_exclusive="2021-01-01",
        )
        forward_wrong_cutoff_ready, _forward_wrong_cutoff_status, _ = validate_expected_r_calibration_artifact(
            calibration_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
            phase_id="forward_oos",
            expected_forward_frozen_cutoff_exclusive="2022-01-01",
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_expected_r_calibration_fits_only_mature_selection_pit_target_and_never_requires_forward_target",
        True,
        calibration_manifest.get("fit_contract", {}).get("forward_target_used_for_fit") is False
        and calibration_manifest.get("fit_contract", {}).get("forward_frozen_cutoff_exclusive") == "2021-01-01"
        and calibration_manifest.get("source_artifacts", {}).get("fit_score_source") == "selection_point_in_time"
        and calibration_manifest.get("fit_rows", [{}])[0].get("sample_count") == 4
        and set(calibration_lookup["group_index"].astype(int)) == {101, 102}
        and calibration_lookup["calibration_cutoff_exclusive"].astype(str).eq("2021-01-01").all()
        and forward_cutoff_ready
        and not forward_wrong_cutoff_ready,
    )

    with tempfile.TemporaryDirectory() as selection_calibration_tmp:
        selection_calibration_root = Path(selection_calibration_tmp)
        selection_score_path = selection_calibration_root / "selection_scores.csv"
        selection_score_path.write_text("selection", encoding="utf-8")
        selection_scores = pd.DataFrame({
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "date": [
                "2013-01-02", "2013-01-02",
                "2014-07-02", "2014-07-02",
                "2015-01-05", "2015-01-05",
            ],
            "group_index": [1, 2, 3, 4, 5, 6],
            "breakout_quality_score": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7],
        })
        selection_contract = SimpleNamespace(score_path=selection_score_path)
        selection_bundle = SimpleNamespace(
            group_table=pd.DataFrame({
                "ticker": ["A", "B", "A", "B", "A", "B"],
                "date": [
                    "2013-01-02", "2013-01-02",
                    "2014-07-02", "2014-07-02",
                    "2015-01-05", "2015-01-05",
                ],
                "group_index": [1, 2, 3, 4, 5, 6],
                "label_eval_end_date": [
                    "2013-02-15", "2013-02-15",
                    "2014-08-15", "2014-08-15",
                    "2015-02-15", "2015-02-15",
                ],
            }),
            raw_target=np.asarray([0.0, 2.0, 0.2, 1.8, 0.4, 1.6], dtype=np.float64),
            target_valid=np.asarray([True] * 6, dtype=bool),
            target_manifest={"target_id": "daily_opportunity_no_time_r_v1"},
        )
        with patch.object(
            expected_r_calibration_service, "_selection_score_frame",
            return_value=(selection_scores, selection_contract),
        ), patch.object(
            expected_r_calibration_service, "_runtime_score_frame",
            return_value=(selection_scores, selection_contract, "selection_point_in_time"),
        ), patch.object(
            expected_r_calibration_service, "load_profile_continuous_ranker_data",
            return_value=selection_bundle,
        ):
            selection_calibration_payload = (
                expected_r_calibration_service.build_expected_r_calibration_artifact(
                    project_root=selection_calibration_root,
                    filter_id="breakout_quality_v1",
                    model_architecture="inception_time_v1",
                    experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
                    phase_id="selection_pit",
                    selection_runtime_start_date="2014-07-01",
                )
            )
        selection_calibration_manifest = dict(selection_calibration_payload["manifest"])
        selection_calibration_lookup = pd.read_csv(selection_calibration_payload["paths"]["lookup"])
        selection_start_ready, _selection_start_status, _ = validate_expected_r_calibration_artifact(
            selection_calibration_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
            phase_id="selection_pit",
            expected_selection_runtime_start_date="2014-07-01",
        )
        selection_wrong_start_ready, _selection_wrong_start_status, _ = validate_expected_r_calibration_artifact(
            selection_calibration_root,
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_v1",
            experiment_profile="daily_universal_no_time_full_list_ndcg_pairwise",
            phase_id="selection_pit",
            expected_selection_runtime_start_date="2014-01-01",
        )
    selection_fit_rows = selection_calibration_manifest.get("fit_rows", [])
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_expected_r_calibration_uses_configured_strategy_start_and_expanding_mature_target_only",
        True,
        selection_calibration_manifest.get("fit_contract", {}).get("selection_runtime_start_date") == "2014-07-01"
        and [row.get("cutoff_exclusive") for row in selection_fit_rows] == ["2014-07-01", "2015-01-01"]
        and [row.get("sample_count") for row in selection_fit_rows] == [2, 4]
        and selection_calibration_lookup["date"].astype(str).min() == "2014-07-02"
        and set(selection_calibration_lookup["group_index"].astype(int)) == {3, 4, 5, 6}
        and selection_start_ready
        and not selection_wrong_start_ready,
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
        "replay_cache": {
            "pairs": {
                arm.arm_id: None
                for arm in settings.enabled_arms
                if arm.dl_enabled
            },
            "baseline_groups": {},
        },
        "dl_sources": {
            str(arm.dl_id): {"ready": True}
            for arm in settings.enabled_arms
            if arm.dl_enabled and arm.dl_id
        },
        "expected_r_calibrations": {
            arm.arm_id: {
                "lookup_path": f"outputs/strategy_compare/runtime_artifacts/{arm.arm_id}/expected_r_lookup.csv.gz"
            }
            for arm in settings.enabled_arms
            if arm.dl_runtime_mode
            == "resource-aware-continuous-expected-pnl-feasible-ascent"
        },
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

    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_completed_pair_cache_contract_checks,
    )
    append_completed_pair_cache_contract_checks(
        results=results,
        case_id=case_id,
        settings=settings,
        ready_status=ready_status,
        mocked_pair_payload=mocked_pair_payload,
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

    from .synthetic_breakout_quality_strategy_preparation_cases import (
        append_strategy_compare_preparation_contract_checks,
    )

    append_strategy_compare_preparation_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
        settings=settings,
        preparation_source=preparation_source,
        orchestration_source=orchestration_source,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_profile_honors_config_and_freezes_min_mr12b_mr13a_runtime_identity",
        True,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_common_period_remains_auto_resolved_from_enabled_dl_sources",
        True,
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
    from config.breakout_quality import get_breakout_quality_workflow_settings

    workflow = get_breakout_quality_workflow_settings()
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
            workflow.experiment_profile,
            "selection_point_in_time",
            workflow.experiment_profile,
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
        "mr13e_multi_seed_profiles_reference_only_enabled_dl_arms_without_mutating_single_seed_roles",
        True,
        all(
            arm_id in selection_ids
            and selection_robust_profile.arms[arm_id].dl_enabled
            and selection_robust_profile.arms[arm_id].robustness_role == "off"
            for arm_id in selection_robust.stochastic_arm_ids
        )
        and all(
            arm_id in forward_ids
            and forward_robust_profile.arms[arm_id].dl_enabled
            and forward_robust_profile.arms[arm_id].robustness_role == "off"
            for arm_id in forward_robust.stochastic_arm_ids
        ),
    )

    summary["selection_arm"] = "C42"
    summary["forward_arm"] = "C44"
    summary["runtime_source"] = "DL-CONT13E"
    summary["pit_source"] = "DL-CONT13E-PIT"
    return results, summary

def validate_breakout_quality_runtime_integration_gate_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_RUNTIME_INTEGRATION_GATE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config.breakout_quality import (
        get_breakout_quality_workflow_settings,
    )
    from config.strategy_compare import get_strategy_runtime_integration_settings
    from core.strategy_comparison import (
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    )
    from filters.breakout_quality.runtime_integration_gate import (
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

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_integration_candidate_is_same_mr13e_exact_semantics_across_stages",
        True,
        (
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
            and dict(forward["dl_runtime_options"]).get("selection_only") is False
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_integration_legacy_gate_is_disabled_after_framework_migration_while_anchor_identity_is_preserved",
        True,
        (
            not cfg.enabled
            and runtime["experiment_profile"] == workflow.experiment_profile
            and cfg.comparison_anchor_experiment_profile != workflow.experiment_profile
            and cfg.selection_profile_id == "selection_pit"
            and cfg.forward_profile_id == "forward_oos"
            and selection["candidate_arm_id"] == cfg.selection_candidate_arm_id
            and forward["candidate_arm_id"] == cfg.forward_candidate_arm_id
        ),
    )
    from filters.breakout_quality.runtime import get_breakout_quality_ranking_source_context

    runtime_context = get_breakout_quality_ranking_source_context()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "promoted_workflow_runtime_uses_mr13e_exact_canonical_context",
        True,
        (
            workflow.runtime_strategy_enabled
            and runtime_context.experiment_profile == workflow.experiment_profile
            and runtime_context.model_architecture == workflow.model_architecture
            and runtime_context.score_source == "canonical_runtime"
            and runtime_context.ranking_policy == workflow.runtime_ranking_policy
            and dict(runtime_context.ranking_options or {}) == dict(workflow.runtime_ranking_options)
        ),
    )
    from filters.breakout_quality.export_scores import parse_args as parse_score_export_args

    parsed_runtime_export = parse_score_export_args(
        [
            "--experiment-profile",
            workflow.experiment_profile,
            "--scope",
            "workflow_runtime",
        ]
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "promoted_mr13e_profile_is_accepted_by_workflow_runtime_score_export_cli",
        (workflow.experiment_profile, "workflow_runtime"),
        (parsed_runtime_export.experiment_profile, parsed_runtime_export.scope),
    )

    from filters.breakout_quality.export_scores import run_export as run_score_export

    with patch(
        "filters.breakout_quality.export_scores._run_daily_continuous_workflow_export",
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "promoted_mr13e_workflow_export_dispatches_to_daily_continuous_path",
        True,
        dispatch_rc == 0 and daily_export.call_count == 1,
    )

    export_source = (
        Path(__file__).resolve().parents[2]
        / "filters" / "breakout_quality" / "export_scores.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "workflow_runtime_scores_are_separate_from_forward_oos_canonical_scores",
        True,
        (
            '"runtime"' in export_source
            and '"runtime_manifest.json"' in export_source
            and "if args.scope == RUNTIME_SCOPE_WORKFLOW" in export_source
            and "canonical Forward-OOS research score/manifest are not modified" in export_source
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "workflow_runtime_mr13e_export_uses_daily_continuous_ranker_contract",
        True,
        (
            "load_daily_universal_ranker_data" in export_source
            and "load_continuous_ranker_oos_contract" in export_source
            and "ranker_api.predict_scores" in export_source
            and '"model_score"' in export_source
        ),
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

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "workflow_runtime_daily_score_lookup_preserves_c44_unavailable_semantics",
        True,
        (
            available_rank["available"] is True
            and available_rank["score"] == 0.75
            and available_rank["score_source"] == "canonical_runtime"
            and missing_rank["available"] is False
            and missing_rank["unavailable_reason"] == "missing_ticker_date_score"
            and missing_rank["score_source"] == "canonical_runtime"
            and outside_coverage_rejected
        ),
    )
    research_source = (Path(__file__).resolve().parents[2] / "apps" / "research.py").read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_promotion_is_explicit_gate_guarded_menu_action",
        True,
        (
            "套用／更新正式 Runtime" in research_source
            and "apply_or_refresh_runtime_promotion" in research_source
            and 'action in {"promote", "apply", "refresh"}' in research_source
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_integration_exact_certificate_uses_exact_active_days_not_max_dl_subset",
        True,
        (
            certificate_status == "PASS"
            and certificate_evidence["certificate_required_days"] == 8
            and certificate_evidence["max_dl_eligible_days_diagnostic_only"] == 5
            and certificate_evidence["certified_days"] == 8
        ),
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_integration_exact_certificate_violation_and_missing_evidence_are_distinct",
        ("FAIL", "BLOCKED"),
        (violation_status, missing_status),
    )

    gate_source = (
        Path(__file__).resolve().parents[2]
        / "filters" / "breakout_quality" / "runtime_integration_gate.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "runtime_integration_reuses_historical_outputs_without_mutating_runtime_default",
        True,
        (
            "settings.reuse_output_roots" in gate_source
            and "workflow.experiment_profile =" not in gate_source
            and "run_runtime_integration_gate" in gate_source
        ),
    )

    summary["candidate_profile"] = runtime["experiment_profile"]
    summary["current_anchor_profile"] = cfg.comparison_anchor_experiment_profile
    summary["promoted_workflow_profile"] = workflow.experiment_profile
    summary["gate_label"] = cfg.label
    return results, summary

