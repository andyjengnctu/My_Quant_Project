from __future__ import annotations

import inspect

from .checks import run_bound_checks

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    Path,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
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
    strategy_app_path = project_root / "apps" / "research.py"
    strategy_config_path = project_root / "config" / "strategy_compare.py"
    research_shell_source = strategy_app_path.read_text(encoding="utf-8")

    check_true(
        "single_seed_contract_has_one_neutral_research_seed_ssot",
        research_source.count("RESEARCH_SINGLE_SEED =") == 1
        and "BREAKOUT_QUALITY_RANDOM_SEED = RESEARCH_SINGLE_SEED" in canonical_source
        and "OPTIMIZER_RANDOM_SEED_DEFAULT = RESEARCH_SINGLE_SEED" in training_policy_source
        and "BREAKOUT_QUALITY_WORKFLOW_RANDOM_SEED" not in canonical_source,
    )

    from config import breakout_quality as breakout_quality_config
    from core import breakout_quality_policy as breakout_quality_policy
    from config import research as research_config
    from config import training_policy as training_policy_config

    configured_seed = breakout_quality_policy.resolve_breakout_quality_random_seed()
    run_bound_checks(
        check_true,
        (
            ('single_seed_contract_resolves_nonnegative_integer', isinstance(configured_seed, int) and configured_seed >= 0,),
            ('single_seed_contract_model_and_optimizer_defaults_share_research_seed', int(breakout_quality_config.BREAKOUT_QUALITY_RANDOM_SEED) == int(training_policy_config.OPTIMIZER_RANDOM_SEED_DEFAULT) == int(research_config.RESEARCH_SINGLE_SEED),),
        ),
    )

    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_RANDOM_SEED", 7):
        overridden_seed = breakout_quality_policy.resolve_breakout_quality_random_seed()
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
        ):
            binary_seed = breakout_quality_policy.get_breakout_quality_workflow_settings().seed
        with patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        ):
            continuous_seed = breakout_quality_policy.get_breakout_quality_workflow_settings().seed
    check(
        "single_seed_contract_applies_override_to_all_profiles",
        (7, 7, 7),
        (overridden_seed, binary_seed, continuous_seed),
    )

    with patch.object(breakout_quality_config, "BREAKOUT_QUALITY_RANDOM_SEED", -1):
        try:
            breakout_quality_policy.resolve_breakout_quality_random_seed()
        except ValueError:
            negative_seed_rejected = True
        else:
            negative_seed_rejected = False
    run_bound_checks(
        check_true,
        (
            ('single_seed_contract_rejects_negative_seed', negative_seed_rejected,),
            (
                'model_and_strategy_apps_are_separate_entries',
                canonical_app_path.is_file()
                and not compatibility_app_path.exists()
                and strategy_app_path.is_file()
                and strategy_config_path.is_file()
                and ('"strategy-compare"' not in canonical_app_source)
                and ('apps/research.py compare' in canonical_app_source)
                and 'if command == "model":' in research_shell_source
                and 'if command in {"compare", "strategy-compare"}:' in research_shell_source
                and '_load_provider_handler("cli_handler")' in research_shell_source
                and 'from services.research.strategy_compare_application import' in research_shell_source,
            ),
        ),
    )

    strategy_compare_source = (
        project_root
        / "services" / "research" / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    strategy_compare_runtime_contract_source = (
        project_root
        / "services" / "research" / "strategy_compare_runtime_contract.py"
    ).read_text(encoding="utf-8")
    from core.research_policy import get_active_model_research_provider

    provider = get_active_model_research_provider()
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

    # Engineering timing diagnostics only: retain every test and every coverage
    # branch. This records which subsection is slow on Windows without profiling
    # all Python calls or changing the 15-second acceptance budget.
    from time import perf_counter
    _section_started = perf_counter()
    _section_durations: list[tuple[str, float]] = []

    def _mark_section(label: str) -> None:
        nonlocal _section_started
        now = perf_counter()
        _section_durations.append((label, now - _section_started))
        _section_started = now

    from core import strategy_compare_policy as strategy_config
    from core.breakout_quality_registry import (
        get_continuous_ranker_research_spec,
    )
    from core.breakout_quality_policy import (
        get_breakout_quality_workflow_settings,
        get_breakout_quality_model_test_settings,
        is_breakout_quality_model_test_profile,
        get_continuous_ranker_execution_recipe,
    )
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_arm_runtime_dl_source_ids,
    )

    # AI: Exercise the real resolver's shared validation path for every registered
    # policy before artifact I/O; source-text assertions cannot detect missing imports.
    from services.research import strategy_compare_runtime_contract as runtime_contract
    for ranking_policy in sorted(runtime_contract.SUPPORTED_BREAKOUT_QUALITY_RANKING_POLICIES):
        with patch.object(runtime_contract, "load_runtime_artifact_contract") as artifact_loader:
            validation_error = None
            try:
                runtime_contract.resolve_strategy_compare_runtime_contract(
                    root=project_root,
                    comparison_mode=runtime_contract.COMPARISON_MODE_HARD_FILTER,
                    ranking_policy=ranking_policy,
                    ranking_options={"stale_score_membership_guard_max_age_days": 0},
                    optional_entry_filter_policy="__invalid_optional_policy__",
                    score_source=runtime_contract.SCORE_SOURCE_CANONICAL_RUNTIME,
                    filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                    model_architecture="validation_fixture",
                    experiment_profile="validation_fixture",
                    threshold=None,
                    hard_filter_source=None,
                )
            except Exception as exc:
                validation_error = exc
            check_true(
                f"runtime_resolver_validates_optional_policy_before_artifact_io:{ranking_policy}",
                isinstance(validation_error, ValueError)
                and "__invalid_optional_policy__" in str(validation_error)
                and not artifact_loader.called,
                note=f"error={type(validation_error).__name__}: {validation_error}",
            )

    # An isolated multi-seed replay must recheck the complete PIT Audit/Fold
    # provenance at execution time. Passing the earlier preparation gate does
    # not authorize an unverified score-file override after the fact.
    from filters.breakout_quality import strategy_compare_diagnostics as pit_diagnostics
    with tempfile.TemporaryDirectory() as isolated_tmp:
        isolated_dir = Path(isolated_tmp)
        isolated_score = isolated_dir / "selection_point_in_time_scores.csv"
        isolated_manifest = isolated_dir / "selection_point_in_time_manifest.json"
        pd.DataFrame({
            "ticker": ["1234"], "date": ["2021-01-04"], "group_index": [1],
            "breakout_quality_score": [0.5], "fold_id": ["fold_1"],
            "model_information_cutoff": ["2020-12-31"],
        }).to_csv(isolated_score, index=False, encoding="utf-8-sig")
        isolated_manifest.write_text(json.dumps({
            "status": "BUILT", "filter_id": "synthetic_filter",
            "model_architecture": "synthetic_arch",
            "experiment_profile": DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "seed": 7,
            "artifacts": {"scores": build_file_manifest(isolated_score)},
            "score_period": {"start": "2021-01-04", "end": "2021-01-04"},
            "coverage": {"scored_group_count": 1},
            "lookahead_contract": {
                "every_score_uses_model_not_trained_on_scored_event": True,
                "training_requires_label_eval_end_before_score_start": True,
                "oos_rows_or_target_statistics_used_for_training_or_epoch_selection": False,
                "future_target_in_score_table": False,
            },
        }), encoding="utf-8")
        isolated_kwargs = {
            "project_root": project_root, "score_path": str(isolated_score),
            "manifest_path": str(isolated_manifest), "filter_id": "synthetic_filter",
            "model_architecture": "synthetic_arch",
            "experiment_profile": DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
            "expected_seed": 7,
        }
        with patch.object(
            pit_diagnostics, "load_selection_point_in_time_ranking_contract",
            side_effect=ValueError("missing verified PIT Audit"),
        ) as canonical_gate:
            try:
                pit_diagnostics._load_isolated_selection_pit_contract(**isolated_kwargs)
            except ValueError as exc:
                missing_audit_rejected = "missing verified PIT Audit" in str(exc)
            else:
                missing_audit_rejected = False
            check_true(
                "isolated_strategy_pit_rechecks_canonical_audit_at_replay",
                missing_audit_rejected and canonical_gate.call_count == 1
                and canonical_gate.call_args.kwargs["require_model_validation_pass"] is False
                and Path(canonical_gate.call_args.kwargs["point_in_time_dir_override"]) == isolated_dir,
            )
        with patch.object(
            pit_diagnostics, "load_selection_point_in_time_ranking_contract",
            return_value=SimpleNamespace(
                score_path=isolated_score, manifest_path=isolated_manifest,
                manifest=json.loads(isolated_manifest.read_text(encoding="utf-8")),
                seed=7,
                model_architecture="synthetic_arch",
                experiment_profile=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
                available_from="2021-01-04",
                available_through="2021-01-04",
            ),
        ):
            isolated_result = pit_diagnostics._load_isolated_selection_pit_contract(
                **isolated_kwargs
            )
            try:
                pit_diagnostics._load_isolated_selection_pit_contract(
                    **{**isolated_kwargs, "expected_seed": 8}
                )
            except ValueError as exc:
                wrong_seed_rejected = "seed" in str(exc)
            else:
                wrong_seed_rejected = False
            try:
                pit_diagnostics._load_isolated_selection_pit_contract(
                    **{**isolated_kwargs, "manifest_path": str(isolated_dir / "wrong_manifest.json")}
                )
            except ValueError as exc:
                wrong_bundle_rejected = "canonical contract" in str(exc)
            else:
                wrong_bundle_rejected = False
            # Model a physical replacement after Ranking/Audit validated the
            # original revision: the consumer must not read a different Score.
            original_score = isolated_score.read_bytes()
            isolated_score.write_bytes(original_score.replace(b"0.5", b"0.8"))
            try:
                pit_diagnostics._load_isolated_selection_pit_contract(**isolated_kwargs)
            except ValueError as exc:
                changed_score_rejected = "SHA256" in str(exc)
            else:
                changed_score_rejected = False
            finally:
                isolated_score.write_bytes(original_score)
        check_true(
            "isolated_strategy_pit_owner_boundary_retains_seed_path_and_physical_revision_guards",
            wrong_seed_rejected and wrong_bundle_rejected and changed_score_rejected,
        )
        check_true(
            "isolated_strategy_pit_retains_approved_seed_and_display_semantics",
            isolated_result.seed == 7
            and isolated_result.available_from == "2021-01-04"
            and isolated_result.available_through == "2021-01-04"
            and isolated_result.model_validation_gate["status"] == "ISOLATED_MULTI_SEED_ROBUSTNESS",
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

    # Completed-result reuse must be registry-derived rather than manually wired
    # per profile.  The current Phase-2 profiles intentionally declare no legacy
    # reuse roots, yet they must still discover the registered Extending namespaces
    # and let canonical result identity decide whether an arm is reusable.
    for settings in settings_by_mode.values():
        registered_result_roots = (
            strategy_config.resolve_strategy_comparison_result_search_roots(settings)
        )
        check_true(
            f"result_reuse_search_is_registry_derived:{settings.profile_id}",
            settings.output_root == registered_result_roots[0]
            and "outputs/strategy_compare/non_dl_broker_cash/oos_2021_forward"
            in registered_result_roots
            and "outputs/strategy_compare/non_dl_broker_cash/rolling_2021_forward"
            in registered_result_roots
            and len(registered_result_roots) == len(set(registered_result_roots)),
        )

    # Historical bug regression: Phase-2 declares no explicit reuse root, yet a
    # scientifically identical C62 completed baseline in the registered Extending
    # namespace must still be discovered. Namespace is not part of reuse legality.
    from core.strategy_comparison import (
        resolve_strategy_comparison_arm_param_policy,
        resolve_strategy_parameter_consumption_mode,
    )
    from filters.breakout_quality.strategy_compare_sources import (
        OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    )
    from filters.breakout_quality.strategy_rule_policies import (
        ALL_RULE_FILTERS_OFF_OVERRIDES,
    )
    from services.research.strategy_compare_reuse import _find_reusable_baseline_source

    phase2_oos = settings_by_mode[
        next(
            str(mode["profile_id"]) for mode in modes
            if bool(mode.get("single_score_block"))
        )
    ]
    check_true(
        "phase2_oos_has_no_manual_cross_profile_reuse_wiring",
        tuple(phase2_oos.reuse_output_roots) == (),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        old_run = (
            temp_root
            / "outputs/strategy_compare/non_dl_broker_cash/oos_2021_forward/runs/old_run"
        )
        old_run.mkdir(parents=True, exist_ok=True)
        (old_run / "strategy_comparison.json").write_text(
            json.dumps({"status": "COMPLETED"}), encoding="utf-8"
        )
        comparison_period = {"start": "2021-01-01", "end": "2026-03-02"}
        resolved_identities = {}
        expected_dirs = {}
        for arm_id in ("C98", "C97", "C96"):
            arm = phase2_oos.arms[arm_id]
            param_sha = f"{arm_id.lower()}-param-eval-sha"
            resolved_identities[arm_id] = {"sha256": param_sha}
            pair_dir = old_run / "pairs" / f"{arm_id.lower()}_baseline"
            pair_dir.mkdir(parents=True, exist_ok=True)
            all_off = arm.rule_policy == "all_off"
            pair_payload = {
                "no_filter": {"total_return_pct": 1.0},
                "metadata": {
                    "dataset": phase2_oos.dataset,
                    "params_file_sha256": param_sha,
                    "requested_param_policy": resolve_strategy_comparison_arm_param_policy(
                        phase2_oos, arm
                    ),
                    "param_evaluation_mode": resolve_strategy_parameter_consumption_mode(
                        phase2_oos.parameter_sources[arm.param_source]
                    ),
                    "optional_entry_filter_policy": (
                        OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
                        if all_off
                        else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
                    ),
                    "shared_param_overrides": (
                        dict(ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {}
                    ),
                    "max_positions": int(phase2_oos.max_positions),
                    "enable_rotation": phase2_oos.rotation == "on",
                    "comparison_period": comparison_period,
                },
            }
            (pair_dir / "strategy_comparison.json").write_text(
                json.dumps(pair_payload), encoding="utf-8"
            )
            for name in (
                "strategy_comparison.md",
                "yearly_returns_comparison.csv",
                "no_filter_equity.csv",
                "no_filter_trades.csv",
                "no_filter_daily_capacity.csv",
                "no_filter_orderable_candidates.csv",
            ):
                (pair_dir / name).write_text("fixture\n", encoding="utf-8")
            expected_dirs[arm_id] = pair_dir

        reuse_status = {
            "comparison_period": comparison_period,
            "resolved_arm_parameter_identities": resolved_identities,
        }
        for arm_id in ("C98", "C97", "C96"):
            reusable = _find_reusable_baseline_source(
                root=temp_root,
                settings=phase2_oos,
                status=reuse_status,
                off_arm=phase2_oos.arms[arm_id],
            )
            check_true(
                f"phase2_reuses_identity_compatible_{arm_id}_from_registered_sibling_profile_without_manual_root",
                reusable == expected_dirs[arm_id],
            )

    _mark_section("runtime_and_archived_baseline")

    shared_models = get_breakout_quality_model_test_settings().model_profiles
    model_bindings = strategy_config.get_strategy_compare_model_bindings()
    check_true(
        "current_strategy_compare_active_membership_is_model_list_driven",
        set(expected_arm_ids).issubset(strategy_config.STRATEGY_COMPARE_ARMS)
        and set(expected_contrast_ids).issubset(strategy_config.STRATEGY_COMPARE_CONTRASTS)
        and tuple((row["model_id"], row["experiment_profile"]) for row in model_bindings) == tuple(shared_models)
        and all((row["status"] == "BOUND") is bool(row["strategy_arm_ids"]) for row in model_bindings),
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
        "research_strategy_menu_has_four_direct_tests_no_one_click_and_runtime_gate_is_main_level",
        'render_menu_item(1, "Forward OOS Test", default=True)' in research_shell_source
        and 'render_menu_item(2, "Rolling OOS Test")' in research_shell_source
        and 'render_menu_item(3, "Forward OOS Robustness Test")' in research_shell_source
        and 'render_menu_item(4, "Rolling OOS Robustness Test")' in research_shell_source
        # AI: Summary moved into the first Research root slot;
        # the four Strategy Compare work types stay [1]..[4] in their submenu.
        and 'render_menu_item(2, get_audit_method("target_ground_truth_strategy_romd").menu_label)' in research_shell_source
        and 'render_menu_item(5, "策略組合比較")' in research_shell_source
        and 'render_menu_item(7, integration_label)' in research_shell_source
        and 'render_menu_item(8, "查看目前研究狀態與工件")' in research_shell_source
        and 'elif choice == "7":\n                _strategy_runtime_integration_menu()' in research_shell_source
        and '"OOS + Rolling 一鍵／Consistency"' not in research_shell_source
        and '"OOS + Rolling Multi-seed 一鍵"' not in research_shell_source
        and "_run_strategy_oos_rolling_one_click" not in research_shell_source
        and "materialize_strategy_oos_rolling_consistency" not in research_shell_source
        and 'elif numeric == 3:\n            _strategy_multi_seed_robustness_menu(str(oos_mode["robustness_id"]))' in research_shell_source
        and 'elif numeric == 4:\n            _strategy_multi_seed_robustness_menu(str(rolling_mode["robustness_id"]))' in research_shell_source,
    )


    oos_settings = next(
        settings_by_mode[str(mode["profile_id"])]
        for mode in modes
        if bool(mode.get("single_score_block"))
    )
    rolling_settings = next(
        settings_by_mode[str(mode["profile_id"])]
        for mode in modes
        if not bool(mode.get("single_score_block"))
    )
    current_rolling_settings = rolling_settings

    # Regression: Forward OOS may freeze one legal schedule member, so its
    # prerequisite producer must not eagerly build later annual refit folds.
    from core.research_orchestration import ResearchArtifactAction
    from core.strategy_comparison import resolve_strategy_parameter_required_schedule_end
    from services.research import strategy_compare_preparation as compare_preparation

    _mark_section("mode_config_and_p5")
    from .synthetic_breakout_quality_strategy_benchmark_cases import (
        append_robustness_benchmark_plan_checks,
    )
    _benchmark_steps, current_robustness = append_robustness_benchmark_plan_checks(
        check=check,
        check_true=check_true,
        project_root=project_root,
        oos_settings=oos_settings,
        rolling_settings=rolling_settings,
        settings_by_mode=settings_by_mode,
        strategy_config=strategy_config,
    )
    _mark_section("benchmark_plan")
    from services.research import strategy_multi_seed_robustness as robustness_plan_runtime
    # AI: Runtime parameter production must enforce the same physical-identity
    # invariant as the preview plan. Shared FIT truth may be produced once, but
    # its consuming arm bindings must not alias the same mutable dictionary.
    synthetic_shared_arms = (
        SimpleNamespace(arm_id="synthetic-A", param_source="shared"),
        SimpleNamespace(arm_id="synthetic-B", param_source="shared"),
    )
    synthetic_shared_robustness = SimpleNamespace(
        benchmark_id="synthetic-binding", resolved_seeds=(17,),
    )

    def _synthetic_runtime_parameter_binding(*, settings, robustness, arm, seed):
        return {
            "arm_id": arm.arm_id, "seed": int(seed),
            "family": "min", "evaluation_mode": "rolling", "param_policy": "min",
            "producer_variant": "canonical_schedule_v1", "training_profile_id": None,
            "fitting_seed_count": 1, "fitting_trials_per_seed": 0,
            "path": Path("isolated") / str(arm.param_source) / "params.json",
            "manifest_path": Path("isolated") / str(arm.param_source) / "manifest.json",
        }

    def _synthetic_runtime_parameter_producer(*, settings, robustness, binding,
                                              seed, comparison_start, comparison_end):
        return {
            **binding, "sha256": "1" * 64, "file_sha256": "2" * 64,
            "manifest_sha256": "3" * 64, "producer_action": "REUSE",
            "parameter_payload_ready": True,
        }

    with (
        patch.object(
            robustness_plan_runtime, "_benchmark_param_binding",
            side_effect=_synthetic_runtime_parameter_binding,
        ),
        patch.object(
            robustness_plan_runtime, "_ensure_benchmark_strategy_parameter_binding",
            side_effect=_synthetic_runtime_parameter_producer,
        ) as shared_producer,
        redirect_stdout(io.StringIO()),
    ):
        runtime_bindings = robustness_plan_runtime._prepare_benchmark_strategy_parameter_artifacts(
            settings=None, robustness=synthetic_shared_robustness,
            benchmark_arms=synthetic_shared_arms,
            comparison_start="2021-01-01", comparison_end="2026-03-02",
        )
    first_binding = runtime_bindings[("synthetic-A", 17)]
    second_binding = runtime_bindings[("synthetic-B", 17)]
    check_true(
        "robustness_runtime_shared_parameter_keeps_independent_arm_bindings",
        shared_producer.call_count == 1
        and first_binding is not second_binding
        and first_binding["arm_id"] == "synthetic-A"
        and second_binding["arm_id"] == "synthetic-B"
        and first_binding["path"] == second_binding["path"]
        and first_binding["sha256"] == second_binding["sha256"],
    )
    conflicting_arms = (
        synthetic_shared_arms[0],
        SimpleNamespace(arm_id="synthetic-B", param_source="different"),
    )
    runtime_conflict_rejected = False
    with (
        patch.object(
            robustness_plan_runtime, "_benchmark_param_binding",
            side_effect=_synthetic_runtime_parameter_binding,
        ),
        patch.object(
            robustness_plan_runtime, "_ensure_benchmark_strategy_parameter_binding",
            side_effect=_synthetic_runtime_parameter_producer,
        ) as conflicting_producer,
        redirect_stdout(io.StringIO()),
    ):
        try:
            robustness_plan_runtime._prepare_benchmark_strategy_parameter_artifacts(
                settings=None, robustness=synthetic_shared_robustness,
                benchmark_arms=conflicting_arms,
                comparison_start="2021-01-01", comparison_end="2026-03-02",
            )
        except ValueError as exc:
            runtime_conflict_rejected = "不得對應不同physical truth" in str(exc)
    check_true(
        "robustness_runtime_rejects_shared_fitting_identity_with_conflicting_paths",
        runtime_conflict_rejected and conflicting_producer.call_count == 1,
    )
    # Regression: current suite capability already resolved C93/raw-price into the
    # same-seed benchmark matrix, but a stale runtime-entry allowlist once rejected
    # every non-canonical data_source before the shared planner was even reached.
    # Exercise the formal entrypoint boundary so config/plan-only tests cannot miss
    # another arm/profile/data-source-specific gate.
    from services.research import strategy_multi_seed_robustness as robustness_runtime

    class _SharedPlannerReached(RuntimeError):
        pass

    runtime_entry_reached_shared_planner = []
    runtime_entry_failures = []
    for robustness_settings in current_robustness:
        with (
            patch.object(robustness_runtime, "collect_artifact_status", return_value={}),
            patch.object(
                robustness_runtime,
                "_render_robustness_execution_plan",
                side_effect=_SharedPlannerReached("shared planner reached"),
            ),
        ):
            try:
                robustness_runtime.run_multi_seed_robustness(
                    robustness_id=robustness_settings.robustness_id,
                    confirm=False,
                )
            except _SharedPlannerReached:
                runtime_entry_reached_shared_planner.append(True)
            except Exception as exc:
                runtime_entry_reached_shared_planner.append(False)
                runtime_entry_failures.append(
                    f"{robustness_settings.robustness_id}: {type(exc).__name__}: {exc}"
                )
            else:
                runtime_entry_reached_shared_planner.append(False)
    check_true(
        "current_robustness_runtime_entry_has_no_arm_profile_or_data_source_gate_before_shared_planner",
        runtime_entry_reached_shared_planner
        and all(runtime_entry_reached_shared_planner),
    )

    robustness_runtime_source = read_source_text(
        "services/research/strategy_multi_seed_robustness.py"
    )
    robustness_runtime_tree = ast.parse(robustness_runtime_source)
    run_multi_seed_node = next(
        node
        for node in robustness_runtime_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_multi_seed_robustness"
    )
    run_multi_seed_source = ast.get_source_segment(
        robustness_runtime_source, run_multi_seed_node
    ) or ""
    check_true(
        "current_robustness_runtime_entry_does_not_own_raw_price_or_c93_authorization",
        "raw_price_arms" not in run_multi_seed_source
        and "raw_unadjusted" not in run_multi_seed_source
        and "C93" not in run_multi_seed_source
        and "Round 1目前只授權" not in run_multi_seed_source,
    )

    # Regression: raw-price benchmark parameter producers must expose the same
    # canonical file/scientific hash contract consumed by the shared robustness
    # binder.  A prior implementation returned READY without scientific_sha256,
    # so training completed and then crashed at the binding boundary.

    # The long-lived extending_current benchmark remains contract-tested even when
    # another research profile is selected as current. Current routing must never
    # redefine its historical/scientific membership.


    # Current authorization is checked against current membership.  The old
    # extending suite remains registered for scientific reproducibility, even
    # when all of its DL conversions have been retired from its active arms.
    enabled_dl_sources = [
        (str(settings.profile_id), str(arm.dl_id), settings.dl_sources[arm.dl_id])
        for settings in (current_rolling_settings,)
        for arm in settings.enabled_arms
        if arm.dl_enabled and arm.dl_id in settings.dl_sources
    ]
    check_true(
        "current_dl_dependencies_use_shared_model_workflow_authorization",
        bool(enabled_dl_sources)
        and all(
            get_breakout_quality_workflow_settings(
                experiment_profile=str(source.experiment_profile)
            ).rolling_authorized
            and is_breakout_quality_model_test_profile(str(source.experiment_profile))
            for _profile_id, _dl_id, source in enabled_dl_sources
        ),
    )

    app_source = read_source_text(Path(__file__).resolve().parents[2] / "apps" / "research.py")
    check_true(
        "research_entry_keeps_generic_strategy_compare_work_type",
        "策略組合比較" in app_source
        and "config/strategy_compare.py" in app_source
        and "MR-13" not in app_source,
    )

    _mark_section("comparison_contract")
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

    _mark_section("preparation_contracts")
    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_completed_pair_score_reuse_contract_checks,
    )
    append_completed_pair_score_reuse_contract_checks(
        results=results,
        case_id=case_id,
        project_root=project_root,
    )
    _mark_section("pair_score_reuse")
    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_model_robustness_pit_strategy_reuse_contract_checks,
    )
    append_model_robustness_pit_strategy_reuse_contract_checks(
        results=results, case_id=case_id,
    )

    _mark_section("model_robustness_contracts")
    from .synthetic_breakout_quality_strategy_reuse_cases import (
        append_completed_pair_baseline_integrity_contract_checks,
    )
    append_completed_pair_baseline_integrity_contract_checks(
        results=results, case_id=case_id
    )

    _mark_section("baseline_integrity")
    from core.training_scheduler import pop_next_seed_diverse_unit
    from services.research import breakout_quality_application as model_application
    from services.research import strategy_compare_training as training_runtime
    from services.research import strategy_multi_seed_robustness as robustness_runtime
    from core.research_policy import get_active_model_research_provider

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

    from .synthetic_breakout_quality_strategy_publication_cases import (
        append_robustness_publication_checks,
        append_latest_publication_checks,
    )
    append_robustness_publication_checks(
        check_true=check_true, project_root=project_root,
        robustness_runtime=robustness_runtime,
    )

    append_latest_publication_checks(check_true=check_true)

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
    run_bound_checks(
        check_true,
        (
            ('dual_model_robustness_keeps_explicit_same_seed_safety_override_over_production_fallback', all(isolated_safety_override_checks) and 'if not safety_override' in execution_contract_source,),
            ('single_and_multi_seed_share_fitting_checkpoint_cache_root', bool(str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT).strip()) and 'multi_seed_robustness' not in str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT) and all((str(strategy_config.get_strategy_multi_seed_robustness_settings(str(mode['robustness_id'])).checkpoint_cache_root) == str(strategy_config.STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT) for mode in modes)) and ('STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT' in application_source),),
        ),
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

    # AI: The historical extending suite can have no current DL arms after
    # retirement; use the active Rolling suite for normal execution coverage.
    rolling_settings = current_rolling_settings
    normal_dl = next(
        current_rolling_settings.dl_sources[arm.dl_id]
        for arm in current_rolling_settings.enabled_arms
        if arm.dl_enabled and arm.dl_id in current_rolling_settings.dl_sources
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
        == [training_runtime.sys.executable, "-m", "services.breakout_quality.point_in_time_scores"]
        and "--resume" in normal_args
        and "--inner-validation-months" in normal_args
        and "--checkpoint-cache-root" in normal_args
        and "--single-score-block" not in normal_args,
    )

    # AI: Historical AH can leave the current Model Compare list without gaining
    # generic PIT/robustness permission; derive the single-seed scope from SSOT.

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

    # AI: All Model/Strategy robustness modes consume the same registered v1
    # seed identity; the independent Target Oracle v2 selection must not change
    # their benchmark. Seed generation still has exactly one training-policy owner.
    from core.training_policy import get_robustness_benchmark_policy_snapshot
    from config.robustness_benchmark import MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID

    benchmark_seed_spec = get_robustness_benchmark_policy_snapshot(
        MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID
    )
    canonical_seeds = tuple(int(seed) for seed in benchmark_seed_spec["resolved_seeds"])
    one_seed_owner = (
        bool(current_robustness)
        and all(
            cfg.benchmark_id == benchmark_seed_spec["benchmark_id"]
            and int(cfg.seed_count) == int(benchmark_seed_spec["seed_count"])
            and int(cfg.seed_generator_seed) == int(benchmark_seed_spec["seed_generator_seed"])
            and tuple(int(seed) for seed in cfg.resolved_seeds) == canonical_seeds
            for cfg in current_robustness
        )
        and all(
            "random.Random(" not in source
            and "def _generate_robustness_benchmark_seeds(" not in source
            for source in (
                read_source_text(project_root / "core" / "strategy_compare_policy.py"),
                robustness_source,
                runtime_gate_source,
            )
        )
    )

    run_bound_checks(
        check_true,
        (
            ('robustness_seed_generation_has_one_training_policy_owner', one_seed_owner,),
            ('single_and_multi_seed_share_arm_model_dependency_resolver', 'def resolve_arm_runtime_dl_source_ids(' in dl_artifact_source and 'def resolve_arm_artifact_dl_source_ids(' in dl_artifact_source and ('resolve_arm_artifact_dl_source_ids(settings, arm)' in dl_artifact_source) and ('resolve_arm_runtime_dl_source_ids(settings, arm)' in robustness_source) and ('def _arm_training_dl_ids(' not in robustness_source),),
            ('strategy_compare_pit_ready_contract_has_one_owner', 'def load_validated_selection_pit_strategy_compare_contract(' in pit_contract_source and 'def validate_selection_pit_score_period(' in pit_contract_source and ('strategy_compare_pit_contract import' in dl_artifact_source) and ('strategy_compare_pit_contract import' in training_contract_source) and ('def _validate_selection_pit_score_period(' not in training_contract_source) and ('load_selection_point_in_time_ranking_contract(' not in training_contract_source),),
            ('pit_fold_schedule_is_shared_by_producer_and_robustness_workload', 'def build_point_in_time_fold_periods(' in pit_schedule_source and 'build_point_in_time_fold_periods' in pit_producer_source and ('build_point_in_time_fold_periods' in robustness_source) and ('def _pit_fold_count_for_period(' not in robustness_source),),
            ('robustness_elapsed_display_reuses_common_formatter', 'from core.display_common import FixedProgressBlock, InlineProgress, format_elapsed' in robustness_source and 'def _format_elapsed(' not in robustness_source,),
        ),
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

    _mark_section("runtime_and_progress")
    summary.update(
        {
            "suite_id": suite_id,
            "mode_count": len(settings_by_mode),
            "arm_count": len(expected_arm_ids),
            "contrast_count": len(expected_contrast_ids),
            "current_dl_profiles": sorted({str(source.experiment_profile) for _profile_id, _dl_id, source in enabled_dl_sources}),
            "runtime_entry_failures": runtime_entry_failures,
        }
    )
    _mark_section("summary")
    # Include the same non-scientific timing breakdown in the machine-readable
    # consistency summary.  The minimal PASS bundle does not include
    # consistency.log, so console-only timing could not diagnose Windows cases.
    summary["section_durations_sec"] = {
        name: round(seconds, 6)
        for name, seconds in (*_section_durations, *_benchmark_steps)
    }
    print(
        "[STRATEGY_COMPARE_CASE_TIMING] "
        + " ".join(f"{name}={seconds:.3f}s" for name, seconds in _section_durations)
    )
    return results, summary



def validate_breakout_quality_daily_pit_strategy_runtime_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_DAILY_PIT_STRATEGY_RUNTIME"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from core import strategy_compare_policy as strategy_config
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
    from services.breakout_quality.point_in_time_scores import (
        _fold_group_ids,
    )

    daily_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
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
            experiment_profile=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
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
        "experiment_profile": DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    })
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            return_value={
                "score": 0.9, "available": True, "score_date": "2020-01-09",
                "score_source": "selection_point_in_time",
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                "model_architecture": "inception_time_v1",
                "experiment_profile": DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE,
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
        profile_name = DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
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
        from filters.breakout_quality.ranker_training_contract import training_semantics
        semantics = training_semantics(daily_profile)
        pairwise_contract = semantics["pairwise_contract"]
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
        manifest["training_semantics"] = semantics
        artifact_paths.manifest_path.write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        load_continuous_ranker_oos_contract.cache_clear()
        load_continuous_ranker_oos_score_table.cache_clear()
        # This fixture exercises score filename/lookup semantics without an
        # upstream Dataset; current-source lineage has its own synthetic Gate.
        with patch(
            "filters.breakout_quality.ranking_score_store._validate_current_forward_score_source_lineage",
            return_value=None,
        ):
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
        from tools.validate.test_forward_publication import independent_upstream_evidence
        from services.breakout_quality.forward_model_source import load_forward_score_binding
        from services.breakout_quality.fitted_model_artifacts import (
            write_fitted_model_contract, current_default_fitting_settings,
        )
        report["standard_model_sop"] = {"evaluation_mode": "forward_oos"}
        (output_dir / "continuous_ranker_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        write_fitted_model_contract(
            model_dir=artifact_paths.model_dir,
            fitting_identity={"training_settings": current_default_fitting_settings()},
            selected_epoch=3, epoch_selection={}, final_refit_history=[],
            trainable_parameter_count=1, total_parameter_count=1,
        )
        with independent_upstream_evidence():
            forward_binding = load_forward_score_binding(dataset="full",
                project_root=root, filter_id=filter_id, model_architecture=architecture,
                experiment_profile=profile_name, expected_seed=42,
                model_dir=artifact_paths.model_dir, research_output_dir=output_dir,
            )
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
                    score_binding=forward_binding,
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

    summary["profile"] = DAILY_UNIVERSAL_NO_TIME_FULL_LIST_NDCG_PAIRWISE_PROFILE
    summary["stage"] = "selection_strategy_translation"
    summary["future_target_used_for_score_presence"] = False
    return results, summary


def validate_breakout_quality_runtime_integration_gate_contract_case(_base_params):
    """Retired producers stay fail-closed while historical evidence remains readable."""
    from services.research.strategy_compare_application import (
        dispatch_runtime_integration_action, runtime_integration_allowed_actions,
        runtime_integration_execution_enabled,
    )
    from services.research.runtime_integration_gate import collect_runtime_integration_status
    results = []
    check, check_true = bind_checks(results, "synthetic_breakout_quality", "RETIRED_RUNTIME_GATE")
    check("retired_gate_never_executes", False, runtime_integration_execution_enabled())
    check("retired_gate_read_only_actions", ("status", "latest"), runtime_integration_allowed_actions())
    check("retired_gate_status_is_blocked", "BLOCKED", collect_runtime_integration_status()["status"])
    for action in ("run", "gate", "promote", "apply", "refresh"):
        try:
            dispatch_runtime_integration_action(action)
        except RuntimeError as exc:
            denied = "退役" in str(exc)
        else:
            denied = False
        check_true(f"retired_gate_rejects_{action}", denied)
    return results, {"ticker": "RETIRED_RUNTIME_GATE", "synthetic": True}

