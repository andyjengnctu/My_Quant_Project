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

def validate_breakout_quality_single_seed_single_entry_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SINGLE_SEED_SINGLE_ENTRY"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    project_root = Path(__file__).resolve().parents[2]
    canonical_config_path = project_root / "config" / "breakout_quality.py"
    canonical_source = canonical_config_path.read_text(encoding="utf-8")
    canonical_app_path = project_root / "services" / "research" / "breakout_quality_application.py"
    compatibility_app_path = project_root / "tools" / "filters" / "breakout_quality" / "application.py"
    canonical_app_source = canonical_app_path.read_text(encoding="utf-8")
    compatibility_app_source = compatibility_app_path.read_text(encoding="utf-8")
    strategy_app_path = project_root / "apps" / "research.py"
    strategy_config_path = project_root / "config" / "strategy_compare.py"

    check(
        "single_seed_contract_has_one_editable_setting_and_no_workflow_seed",
        (1, False),
        (
                    canonical_source.count("BREAKOUT_QUALITY_RANDOM_SEED ="),
                    "BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED" in canonical_source,
                ),
    )

    from config import breakout_quality as breakout_quality_config

    configured_seed = breakout_quality_config.resolve_breakout_quality_random_seed()
    check_true(
        "single_seed_contract_resolves_nonnegative_integer",
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
        / "filters"
        / "breakout_quality"
        / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    from config.research import get_active_model_research_provider

    provider = get_active_model_research_provider()
    research_shell_source = strategy_app_path.read_text(encoding="utf-8")
    check_true(
        "research_formal_applications_are_service_owned_and_tools_are_compatibility_only",
        provider.module == "services.research.breakout_quality_application"
        and "from services.optimizer.application import main" in research_shell_source
        and "from services.audit.runner import" in research_shell_source
        and "from tools.optimizer" not in research_shell_source
        and "from tools.audit" not in research_shell_source,
    )

    check_true(
        "single_seed_contract_strategy_gate_rejects_seed_mismatch",
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
    """Cross-domain Strategy Compare integration contract.

    Detailed preparation, cache/reuse, selector math, PIT, reporting and artifact
    contracts live in their dedicated validators.  This case intentionally owns
    only the integration invariants that join those domains together.
    """
    case_id = "STRATEGY_COMPARE_CONFIG_DRIVEN_APP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

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
        == {"CONT13E_ROLL", "CONT13K_ROLL", "CONT13M_ROLL"}
        and set(strategy_config.STRATEGY_COMPARE_ARMS) == set(expected_arm_ids)
        and set(strategy_config.STRATEGY_COMPARE_CONTRASTS) == set(expected_contrast_ids)
        and {"full_roos", "min_roos", "selection_min_roos", "selection_full_roos"}
        .issubset(set(strategy_history.HISTORICAL_STRATEGY_PARAM_SOURCES))
        and {"CONT13E", "CONT13E_PIT", "CONT13K", "CONT13M", "CONT13M_PIT", "CONT13K_PIT"}
        .issubset(set(strategy_history.HISTORICAL_STRATEGY_DL_SOURCES))
        and {"C1", "C3", "C23", "C32", "C42", "C44", "C56", "C57"}
        .issubset(set(strategy_history.HISTORICAL_STRATEGY_COMPARE_ARMS)),
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
                policy="base-finalist-best",
            )
            == resolve_strategy_param_benchmark_artifact_path(
                Path(__file__).resolve().parents[2],
                benchmark_id=str(benchmark["benchmark_id"]),
                seed=seed,
                family=family,
                evaluation_mode="rolling",
                policy="base-finalist-best",
            )
            for family in STRATEGY_PARAM_FAMILIES
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
    project_root = Path(__file__).resolve().parents[2]
    from .synthetic_breakout_quality_strategy_preparation_cases import (
        append_strategy_compare_preparation_contract_checks,
    )
    append_strategy_compare_preparation_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
        settings=settings,
        preparation_source=read_source_text(
            project_root / "filters" / "breakout_quality" / "strategy_compare_preparation.py"
        ),
        orchestration_source=read_source_text(
            project_root / "filters" / "breakout_quality" / "strategy_comparison.py"
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
    from filters.breakout_quality import strategy_multi_seed_robustness as robustness_runtime

    application_source = read_source_text(
        project_root / "services" / "research" / "breakout_quality_application.py"
    )
    robustness_source = read_source_text(
        project_root / "filters" / "breakout_quality" / "strategy_multi_seed_robustness.py"
    )
    check_true(
        "normal_and_robustness_share_gpu_training_worker_ssot_and_seed_diverse_scheduler",
        int(strategy_config.STRATEGY_COMPARE_GPU_TRAIN_WORKERS) >= 1
        and float(strategy_config.STRATEGY_COMPARE_TRAIN_PROGRESS_INTERVAL_SECONDS) > 0.0
        and all(
            int(strategy_config.get_strategy_multi_seed_robustness_settings(str(mode["robustness_id"])).gpu_train_workers)
            == int(strategy_config.STRATEGY_COMPARE_GPU_TRAIN_WORKERS)
            for mode in modes
        )
        and "ThreadPoolExecutor(max_workers=worker_count)" in application_source
        and "subprocess.Popen(" in application_source
        and "pop_next_seed_diverse_unit(jobs, futures.values())" in application_source
        and "pop_next_seed_diverse_unit(" in robustness_source,
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
    normal_command, normal_args, _ = model_application._strategy_compare_model_build_command(
        source=normal_dl,
        workflow=normal_workflow,
        pit_dir_override=None,
    )
    seed_arg_index = normal_args.index("--seed") + 1
    check_true(
        "normal_parallel_training_keeps_canonical_workflow_seed_and_isolated_subprocess",
        int(normal_args[seed_arg_index]) == int(normal_workflow.seed)
        and normal_command[:3]
        == [model_application.sys.executable, "-m", "tools.filters.breakout_quality.build_point_in_time_scores"]
        and "--resume" in normal_args
        and "--single-score-block" not in normal_args,
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
    fixed_progress = robustness_runtime._FixedProgressBlock(stream=tty_buffer)
    fixed_progress.update(["seed-1 first", "seed-2 first"])
    first_render_len = len(tty_buffer.parts)
    fixed_progress.update(["seed-1 next", "seed-2 next"])
    second_render = "".join(tty_buffer.parts[first_render_len:])
    fixed_progress.clear()
    check_true(
        "robustness_training_progress_redraws_fixed_seed_lines_in_place",
        "\x1b[1A" in second_render
        and "seed-1 next" in second_render
        and "seed-2 next" in second_render,
    )

    with tempfile.TemporaryDirectory(prefix="robustness_atomic_retry_") as temp_dir:
        target = Path(temp_dir) / "manifest.json"
        target.write_text("old\n", encoding="utf-8")
        original_replace = robustness_runtime.os.replace
        replace_attempts = {"count": 0}

        def transient_replace(src, dst):
            replace_attempts["count"] += 1
            if replace_attempts["count"] <= 2:
                raise PermissionError("simulated transient manifest lock")
            return original_replace(src, dst)

        with patch.object(robustness_runtime.os, "replace", side_effect=transient_replace), patch.object(
            robustness_runtime.time, "sleep", return_value=None
        ):
            robustness_runtime._atomic_write_text(target, "new\n")

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
        "runtime_integration_legacy_gate_is_disabled_after_framework_migration_while_anchor_identity_is_preserved",
        not cfg.enabled
                    and runtime["experiment_profile"] == workflow.experiment_profile
                    and cfg.comparison_anchor_experiment_profile != workflow.experiment_profile
                    and cfg.selection_profile_id == "selection_pit"
                    and cfg.forward_profile_id == "forward_oos"
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
    from filters.breakout_quality.export_scores import parse_args as parse_score_export_args

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
    check_true(
        "promoted_mr13e_workflow_export_dispatches_to_daily_continuous_path",
        dispatch_rc == 0 and daily_export.call_count == 1,
    )

    export_source = (
        Path(__file__).resolve().parents[2]
        / "filters" / "breakout_quality" / "export_scores.py"
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
    research_source = (Path(__file__).resolve().parents[2] / "apps" / "research.py").read_text(encoding="utf-8")
    check_true(
        "runtime_promotion_is_explicit_gate_guarded_menu_action",
        "套用／更新正式 Runtime" in research_source
                    and "apply_or_refresh_runtime_promotion" in research_source
                    and 'action in {"promote", "apply", "refresh"}' in research_source,
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
        / "filters" / "breakout_quality" / "runtime_integration_gate.py"
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

