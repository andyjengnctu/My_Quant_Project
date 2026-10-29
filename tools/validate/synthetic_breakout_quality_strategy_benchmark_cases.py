"""Config-driven Robustness benchmark planning integration checks.

This capability helper is called by the existing Strategy Compare synthetic case;
it does not register a second test entry or establish scientific identities.
"""

from __future__ import annotations

from time import perf_counter
from .synthetic_breakout_quality_support import Path, SimpleNamespace, patch, pd, tempfile


def append_robustness_benchmark_plan_checks(
    *, check, check_true, project_root, oos_settings, rolling_settings,
    settings_by_mode, strategy_config,
) -> list[tuple[str, float]]:
    from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
    from core.strategy_comparison import resolve_strategy_comparison_arm_param_policy
    from core.training_policy import (
        get_robustness_benchmark_policy_snapshot,
        resolve_robustness_benchmark_seeds,
    )
    from core.strategy_param_artifacts import (
        STRATEGY_PARAM_FAMILIES,
        resolve_strategy_param_artifact_path,
        resolve_strategy_param_benchmark_artifact_path,
    )
    from services.research.strategy_compare_application import (
        build_strategy_oos_rolling_consistency,
    )

    # AI: Benchmark subsections remain within the existing 15-second case budget.
    # They are diagnostics, not separate tests or a relaxed performance gate.
    _benchmark_started = perf_counter()
    _benchmark_steps: list[tuple[str, float]] = []

    def _mark_benchmark_step(label: str) -> None:
        nonlocal _benchmark_started
        now = perf_counter()
        _benchmark_steps.append((f"benchmark_{label}", now - _benchmark_started))
        _benchmark_started = now

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
            tuple(resolve_robustness_benchmark_seeds()),
        ),
        (
            int(benchmark["strategy_trials_per_fold"]),
            tuple(benchmark["resolved_seeds"]),
        ),
    )

    # AI: The benchmark registry is config-driven, but an existing benchmark_id is immutable.
    # Isolated overrides remain legal for tests; changing the registered spec under the same ID
    # must fail before runtime can silently reinterpret historical scientific evidence.
    from config import robustness_benchmark as training_policy_config
    current_benchmark_id = str(training_policy_config.ROBUSTNESS_BENCHMARK_ID)
    current_benchmark_spec = dict(
        training_policy_config.ROBUSTNESS_BENCHMARK_SPECS[current_benchmark_id]
    )
    check_true(
        "robustness_benchmark_current_spec_is_registered_and_resolves_exactly",
        int(current_benchmark_spec["seed_count"]) == int(benchmark["seed_count"])
        and int(current_benchmark_spec["seed_generator_seed"])
        == int(benchmark["seed_generator_seed"])
        and int(benchmark["seed_count"]) == len(tuple(benchmark["resolved_seeds"]))
        and len(set(int(value) for value in benchmark["resolved_seeds"]))
        == int(benchmark["seed_count"]),
    )
    drifted_specs = {
        str(key): dict(value)
        for key, value in training_policy_config.ROBUSTNESS_BENCHMARK_SPECS.items()
    }
    drifted_specs[current_benchmark_id] = {
        **current_benchmark_spec,
        "seed_count": int(current_benchmark_spec["seed_count"]) + 1,
    }
    with patch.object(
        training_policy_config, "ROBUSTNESS_BENCHMARK_SPECS", drifted_specs
    ):
        try:
            resolve_robustness_benchmark_seeds()
        except ValueError as exc:
            benchmark_drift_error = str(exc)
        else:
            benchmark_drift_error = ""
    check_true(
        "robustness_benchmark_same_id_scientific_drift_fails_fast",
        "scientific identity drift" in benchmark_drift_error
        and "新的benchmark_id" in benchmark_drift_error,
        note=benchmark_drift_error,
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
    robustness_suite_ids = {item.suite_id for item in current_robustness}
    robustness_suite_id = (
        next(iter(robustness_suite_ids)) if len(robustness_suite_ids) == 1 else None
    )
    robustness_suite = (
        strategy_config.get_strategy_compare_suite(str(robustness_suite_id))
        if robustness_suite_id
        else None
    )
    robustness_expected_arm_ids = (
        tuple(robustness_suite["arm_ids"]) if robustness_suite is not None else ()
    )
    current_compare_suite_ids = {
        strategy_config.get_strategy_comparison_settings(str(mode["profile_id"])).suite_id
        for mode in strategy_config.get_strategy_rolling_test_modes()
    }
    check_true(
        "current_robustness_routes_extend_the_same_current_compare_suite",
        bool(robustness_suite_id)
        and current_compare_suite_ids == {robustness_suite_id}
        and all(
            item.enabled
            and item.suite_id == robustness_suite_id
            and strategy_config.get_strategy_comparison_settings(item.profile_id).suite_id
            == item.suite_id
            and not item.fixed_arm_ids
            and tuple(item.stochastic_arm_ids) == robustness_expected_arm_ids
            for item in current_robustness
        ),
    )

    _mark_benchmark_step("policy_and_seed_matrix")
    from core.strategy_compare_policy import resolve_strategy_robustness_result_search_roots
    from services.research.strategy_multi_seed_robustness import (
        _benchmark_param_binding,
        _benchmark_parameter_plan_rows,
    )
    representative_fixed_bindings = {}
    for robustness in current_robustness:
        roots = resolve_strategy_robustness_result_search_roots(robustness)
        check_true(
            f"robustness_result_reuse_search_is_registry_derived:{robustness.robustness_id}",
            roots[0] == robustness.output_root
            and "outputs/strategy_compare/robustness/extending_window/oos_2021_forward" in roots
            and "outputs/strategy_compare/robustness/extending_window/rolling_2021_forward" in roots
            and len(roots) == len(set(roots)),
        )

        robust_settings = strategy_config.get_strategy_comparison_settings(robustness.profile_id)
        canonical_reuse_arm_ids = tuple(
            arm_id
            for arm_id in robustness.stochastic_arm_ids
            if (
                robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder is not None
                and str(
                    robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder.builder_type
                ) == "canonical_optimizer_strategy_params"
            )
        )
        fixed_multiseed_arm_ids = tuple(
            arm_id
            for arm_id in robustness.stochastic_arm_ids
            if (
                robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder is not None
                and str(
                    robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder.builder_type
                ) == "optimizer_fixed_multiseed_strategy_params"
            )
        )
        raw_price_arm_ids = tuple(
            arm_id
            for arm_id in robustness.stochastic_arm_ids
            if (
                robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder is not None
                and str(
                    robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder.builder_type
                ) == "optimizer_raw_price_strategy_params"
            )
        )
        raw_price_profile_ids = {
            str(dict(robust_settings.parameter_sources[str(robust_settings.arms[arm_id].param_source)].builder.options).get("training_profile_id") or "")
            for arm_id in raw_price_arm_ids
        }
        from core.strategy_param_policy import (
            get_strategy_param_fitting_seed_protocol,
            get_strategy_param_training_profile,
        )
        fixed_profile_variants = {
            str(profile_id): (
                "fixed_multiseed_v1"
                if get_strategy_param_fitting_seed_protocol(
                    get_strategy_param_training_profile(str(profile_id)).fitting_seed_protocol
                ).enabled else "fixed_single_seed_v1"
            )
            for arm_id in fixed_multiseed_arm_ids
            for profile_id in (
                dict(robust_settings.parameter_sources[
                    str(robust_settings.arms[arm_id].param_source)
                ].builder.options).get("training_profile_id"),
            )
            if profile_id
        }
        reusable = {
            (arm_id, int(seed)): {"source_run_root": Path("synthetic_registered_run")}
            for arm_id in canonical_reuse_arm_ids
            for seed in robustness.resolved_seeds
        }
        benchmark_arms = tuple(
            robust_settings.arms[arm_id] for arm_id in robustness.stochastic_arm_ids
        )
        from services.research import strategy_multi_seed_robustness as robustness_owner
        # AI: This test owns the seed/source -> dependency-plan contract, not the
        # optimizer's artifact inspection. The latter is exercised with real
        # temporary files, registration drift and BUILD/REUSE in the dedicated
        # Strategy Parameter validators. Avoid probing the user's actual
        # historical optimizer workspace once per seed when checking dedup.
        with (
            patch.object(
                robustness_owner, "_benchmark_param_binding",
                wraps=robustness_owner._benchmark_param_binding,
            ) as binding_resolver,
            patch.object(
                robustness_owner,
                "resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
                return_value={"action": "BUILD", "status": "ARTIFACT_MISSING_OR_INVALID"},
            ) as fixed_inspector,
        ):
            rows, blockers, _bindings = _benchmark_parameter_plan_rows(
                settings=robust_settings,
                robustness=robustness,
                benchmark_arms=benchmark_arms,
                comparison_start="2021-01-01",
                comparison_end="2026-03-02",
                reusable_observations=reusable,
            )
        unique_source_bindings = {
            (int(seed), str(arm.param_source), str(
                resolve_strategy_comparison_arm_param_policy(robust_settings, arm)
            ))
            for seed in robustness.resolved_seeds
            for arm in benchmark_arms
        }
        check_true(
            f"robustness_param_binding_is_resolved_once_per_shared_source:{robustness.robustness_id}",
            binding_resolver.call_count == len(unique_source_bindings),
        )
        reused_result_rows = [
            row for row in rows
            if row[0] == "REUSE" and str(row[1]).startswith("result:")
        ]
        # One physical parameter truth may serve multiple DL arms with different
        # PIT scores and allocators.  Each seed×physical truth must be scheduled
        # only once even though arm×seed replay observations stay independent.
        parameter_dependency_rows = [
            row for row in rows if str(row[1]).startswith("param-benchmark:")
        ]
        reused_keys = {
            (arm_id, int(seed))
            for arm_id in canonical_reuse_arm_ids
            for seed in robustness.resolved_seeds
        }
        distinct_param_truth_paths = {
            (str(binding["path"]), str(binding["manifest_path"]))
            for arm_seed, binding in _bindings.items()
            if arm_seed not in reused_keys
        }
        # AI: Keep every unique fixed-multiseed seed/unit routed through the
        # canonical inspector; only its filesystem-backed result is isolated.
        expected_fixed_units = {
            (int(binding["seed"]), str(binding["training_profile_id"]),
             str(binding["param_policy"]))
            for unit, binding in _bindings.items()
            if unit not in reused_keys
            and str(binding["producer_variant"]) in {"fixed_multiseed_v1", "fixed_single_seed_v1"}
        }
        for unit, binding in _bindings.items():
            if unit not in reused_keys and str(binding["producer_variant"]) in {
                "fixed_multiseed_v1", "fixed_single_seed_v1"
            }:
                representative_fixed_bindings.setdefault(
                    (str(binding["training_profile_id"]), str(binding["param_policy"])),
                    (robustness, robust_settings, binding),
                )
        observed_fixed_units = {
            (int(call.kwargs["replication_seed"]),
             str(call.kwargs["training_profile_id"]), str(call.kwargs["policy"]))
            for call in fixed_inspector.call_args_list
        }
        check_true(
            f"robustness_fixed_seed_inspector_is_called_once_per_physical_unit:{robustness.robustness_id}",
            len(fixed_inspector.call_args_list) == len(expected_fixed_units)
            and observed_fixed_units == expected_fixed_units
            and all(
                str(call.kwargs["benchmark_id"]) == str(robustness.benchmark_id)
                and str(call.kwargs["dataset"]) == str(robust_settings.dataset)
                and str(call.kwargs["rotation"]) == str(robust_settings.rotation)
                and int(call.kwargs["max_positions"]) == int(robust_settings.max_positions)
                for call in fixed_inspector.call_args_list
            ),
        )
        check_true(
            f"robustness_reused_scientific_units_prune_param_dependencies:{robustness.robustness_id}",
            not blockers
            and len(reused_result_rows) == len(canonical_reuse_arm_ids) * len(robustness.resolved_seeds)
            and all(
                any(f"result:{arm_id}:seed={int(seed)}" == row[1] for row in reused_result_rows)
                for arm_id in canonical_reuse_arm_ids
                for seed in robustness.resolved_seeds
            )
            and len(parameter_dependency_rows) == len(distinct_param_truth_paths)
            and all(
                str(binding.get("plan_action") or "")
                in {"REUSE", "BUILD", "REBUILD", "REPAIR", "CHECK"}
                for arm_seed, binding in _bindings.items()
                if arm_seed not in reused_keys
            )
            and all(
                len({
                    str(binding.get("plan_action") or "")
                    for arm_seed, binding in _bindings.items()
                    if arm_seed not in reused_keys
                    and str(binding["path"]) == path
                    and str(binding["manifest_path"]) == manifest
                }) == 1
                for path, manifest in distinct_param_truth_paths
            )
            and len({str(row[1]) for row in parameter_dependency_rows})
            == len(parameter_dependency_rows)
            and all(
                (
                    any(
                        ("replicate=" if variant == "fixed_multiseed_v1" else "seed=")
                            in str(row[1])
                        and f":{variant}:" in str(row[1])
                        and f":profile={profile_id}" in str(row[1])
                        for profile_id, variant in fixed_profile_variants.items()
                    )
                )
                or (
                    ":seed=" in str(row[1])
                    and "raw_price_isolated_v1" in str(row[1])
                    and any(
                        f":profile={profile_id}" in str(row[1])
                        for profile_id in raw_price_profile_ids
                        if profile_id
                    )
                )
                for row in parameter_dependency_rows
            ),
        )

    # AI: One real, isolated filesystem-backed inspection per distinct registered
    # profile/policy preserves the scientific-identity and missing-artifact
    # branches here. Per-seed production/reuse/tamper cases remain the job of
    # the dedicated Strategy Parameter tests, never the user's live workspace.
    with tempfile.TemporaryDirectory(prefix="benchmark_plan_inspector_") as temp_dir:
        for (profile_id, policy), (robustness, robust_settings, binding) in sorted(
            representative_fixed_bindings.items()
        ):
            seed = int(binding["seed"])
            state = robustness_owner.resolve_fixed_multiseed_robustness_strategy_parameter_artifact(
                temp_dir,
                training_profile_id=profile_id,
                benchmark_id=str(robustness.benchmark_id),
                replication_seed=seed,
                policy=policy,
                comparison_end_date="2026-03-02",
                dataset=str(robust_settings.dataset),
                max_positions=int(robust_settings.max_positions),
                rotation=str(robust_settings.rotation),
                fixed_risk=float(robustness_owner.DEFAULT_FIXED_RISK),
                max_position_cap_pct=float(robustness_owner.DEFAULT_MAX_POSITION_CAP_PCT),
            )
            check_true(
                f"robustness_fixed_inspector_registered_identity:{profile_id}:{policy}",
                state.get("action") == "BUILD"
                and state.get("status") == "ARTIFACT_MISSING_OR_INVALID"
                and bool(state.get("scientific_identity", {}).get("source_registration_matches"))
                and int(state.get("build_contract", {}).get("robustness_replication_seed", -1)) == seed,
            )

    _mark_benchmark_step("source_binding_plans")
    from services.research.strategy_multi_seed_robustness import (
        _render_robustness_execution_plan,
        _robustness_arms,
        collect_artifact_status as collect_robustness_artifact_status,
    )
    rolling_robustness = next(
        item for item in current_robustness if item.profile_id.endswith("_rolling")
    )
    rolling_robust_settings = strategy_config.get_strategy_comparison_settings(
        rolling_robustness.profile_id
    )
    rolling_fixed, rolling_stochastic = _robustness_arms(
        rolling_robust_settings, rolling_robustness
    )
    rolling_canonical_reuse_arm_ids = tuple(
        arm_id
        for arm_id in rolling_robustness.stochastic_arm_ids
        if (
            rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder is not None
            and str(
                rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder.builder_type
            ) == "canonical_optimizer_strategy_params"
        )
    )
    rolling_fixed_multiseed_arm_ids = tuple(
        arm_id
        for arm_id in rolling_robustness.stochastic_arm_ids
        if (
            rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder is not None
            and str(
                rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder.builder_type
            ) == "optimizer_fixed_multiseed_strategy_params"
        )
    )
    rolling_raw_price_arm_ids = tuple(
        arm_id
        for arm_id in rolling_robustness.stochastic_arm_ids
        if (
            rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder is not None
            and str(
                rolling_robust_settings.parameter_sources[str(rolling_robust_settings.arms[arm_id].param_source)].builder.builder_type
            ) == "optimizer_raw_price_strategy_params"
        )
    )
    from services.research import strategy_multi_seed_robustness as robustness_plan_runtime

    rolling_reusable = {
        (arm_id, int(seed)): {
            "source_run_root": Path("synthetic_registered_extending_run"),
            "row": {
                "strategy_param_sha256": "synthetic",
                "strategy_param_manifest_sha256": "synthetic-manifest",
            },
            "yearly": pd.DataFrame(),
        }
        for arm_id in rolling_canonical_reuse_arm_ids
        for seed in rolling_robustness.resolved_seeds
    }
    # AI: The plan must exercise the canonical preparation/model-score inspectors,
    # but a synthetic planning invariant cannot depend on the size, age or validity
    # of an operator's physical research artifacts.  Inspect a fresh isolated root.
    with (
        tempfile.TemporaryDirectory(prefix="strategy_compare_rolling_plan_") as isolated_root,
        patch.object(robustness_plan_runtime, "PROJECT_ROOT", Path(isolated_root)),
        patch.object(
            robustness_plan_runtime,
            "inspect_reusable_model_research_rolling_score",
            wraps=robustness_plan_runtime.inspect_reusable_model_research_rolling_score,
        ) as isolated_model_inspections,
        patch(
            "services.research.strategy_multi_seed_robustness._discover_registered_strategy_only_scientific_reuse",
            return_value=rolling_reusable,
        ),
        # This fixture explicitly tests the plan before fixed-multiseed work is
        # prepared; isolate it from parameter artifacts already present on disk.
        patch(
            "services.research.strategy_multi_seed_robustness.resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
            return_value={"action": "BUILD"},
        ),
        patch(
            "services.research.strategy_multi_seed_robustness.inspect_raw_price_robustness_strategy_parameter_artifact",
            return_value={"action": "BUILD", "status": "ARTIFACT_MISSING"},
        ),
    ):
        isolated_status_started = perf_counter()
        isolated_plan_status = collect_robustness_artifact_status(
            project_root=Path(isolated_root), settings=rolling_robust_settings,
        )
        isolated_status_seconds = perf_counter() - isolated_status_started
        # Exercise the real dependency-pending branch as well; a missing upstream
        # cannot authorize a completed scientific result or Model Score REUSE.
        _, pending_rolling_plan = _render_robustness_execution_plan(
            cfg=rolling_robustness,
            settings=rolling_robust_settings,
            status=isolated_plan_status,
            fixed_arms=rolling_fixed,
            stochastic_arms=rolling_stochastic,
        )
        # This test's first rolling scenario is a post-upstream planning snapshot:
        # the canonical isolated preparation observer has already been called,
        # while the producer-completion stage is represented by explicit fixture
        # state. It must not query real project artifacts for this synthetic check.
        first_rolling_plan_status = {
            **isolated_plan_status,
            "preparation_plan": SimpleNamespace(actions=[]),
        }
        isolated_render_started = perf_counter()
        rendered_robustness_plan, rendered_robustness_payload = (
            _render_robustness_execution_plan(
                cfg=rolling_robustness,
                settings=rolling_robust_settings,
                status=first_rolling_plan_status,
                fixed_arms=rolling_fixed,
                stochastic_arms=rolling_stochastic,
            )
        )
        isolated_render_seconds = perf_counter() - isolated_render_started
    check_true(
        "robustness_first_rolling_plan_does_not_reuse_results_while_upstream_is_missing",
        bool(pending_rolling_plan.get("model_upstream_prepare_required"))
        and pending_rolling_plan.get("overall_status") in {"PREPARABLE", "BLOCKED"}
        and not pending_rolling_plan.get("registered_scientific_reuse_units"),
    )
    check_true(
        "robustness_first_rolling_plan_real_isolated_preparation_observer_detects_missing_inputs",
        bool(isolated_plan_status.get("preparation_plan"))
        and any(
            action.action in {"BUILD", "REBUILD", "RESUME"}
            for action in isolated_plan_status["preparation_plan"].actions
        ),
    )
    # A completed Strategy Result avoids redundant Model Score inspection.
    # For every still-pending model source the canonical inspector must receive
    # the exact registered seed and the isolated root, not the user's workspace.
    model_seed_arms = robustness_plan_runtime._model_seed_sensitive_arms(
        rolling_robust_settings, rolling_robustness, tuple(rolling_stochastic),
    )
    expected_model_inspections = {
        (
            str(rolling_robust_settings.dl_sources[str(dl_id)].experiment_profile),
            int(seed),
        )
        for dl_id, arm_entries in robustness_plan_runtime._training_source_groups(
            tuple(model_seed_arms), settings=rolling_robust_settings,
        )
        for seed in rolling_robustness.resolved_seeds
        if not all(
            (arm.arm_id, int(seed)) in rolling_reusable
            for _order, arm in arm_entries
        )
    }
    observed_model_inspections = [
        (str(call.kwargs["source"].experiment_profile), int(call.kwargs["seed"]))
        for call in isolated_model_inspections.call_args_list
    ]
    check_true(
        "robustness_first_rolling_plan_inspects_each_model_seed_without_workspace_artifacts",
        bool(expected_model_inspections)
        and len(observed_model_inspections) == len(expected_model_inspections)
        and set(observed_model_inspections) == expected_model_inspections
        and all(
            Path(call.kwargs["project_root"]) == Path(isolated_root)
            for call in isolated_model_inspections.call_args_list
        ),
        note=(
            f"expected_units={len(expected_model_inspections)}, "
            f"observed_units={len(observed_model_inspections)}"
        ),
    )
    expected_reuse_units = len(rolling_canonical_reuse_arm_ids) * len(rolling_robustness.resolved_seeds)
    rolling_pending_param_truth_paths = {
        (str(binding["path"]), str(binding["manifest_path"]))
        for arm_id in (*rolling_fixed_multiseed_arm_ids, *rolling_raw_price_arm_ids)
        for seed in rolling_robustness.resolved_seeds
        for binding in (
            _benchmark_param_binding(
                settings=rolling_robust_settings,
                robustness=rolling_robustness,
                arm=rolling_robust_settings.arms[arm_id],
                seed=int(seed),
            ),
        )
    }
    expected_pending_search_units = len(rolling_pending_param_truth_paths)
    expected_total_units = len(rolling_robustness.stochastic_arm_ids) * len(rolling_robustness.resolved_seeds)
    check_true(
        "robustness_plan_surfaces_cross_profile_seed_reuse_before_pending_fixed_multiseed_work",
        len(rendered_robustness_payload.get("registered_scientific_reuse_units") or ()) == expected_reuse_units
        and f"{expected_pending_search_units} pending search units" in rendered_robustness_plan
        and f"scientific result REUSE={expected_reuse_units}/{expected_total_units} arm×replicate" in rendered_robustness_plan
        and rendered_robustness_plan.count("PARAM+REPLAY") == sum(
            1 for arm in rolling_stochastic
            if not arm.dl_enabled and arm.arm_id not in rolling_canonical_reuse_arm_ids
        )
        and rendered_robustness_plan.count(" seed scientific results identity一致") == len(rolling_canonical_reuse_arm_ids)
        and "Strategy fitting budgets" in rendered_robustness_plan
        and "4 seeds x 300 trials" in rendered_robustness_plan
        and "8 seeds x 300 trials" not in rendered_robustness_plan
        and "4 seeds x 1000 trials" not in rendered_robustness_plan,
    )

    _mark_benchmark_step("first_rolling_plan")
    # Additional engineering diagnostics; the original first_rolling_plan subtotal
    # and the single 15-second test-case budget stay unchanged.
    _benchmark_steps.extend((
        ("benchmark_first_rolling_preparation_status", isolated_status_seconds),
        ("benchmark_first_rolling_renderer", isolated_render_seconds),
    ))
    # Previously fitted robustness parameter artifacts are replay prerequisites, not
    # pending Optimizer searches.  When their full scientific contracts are READY,
    # the plan must show zero pending search even if no completed replay result exists.
    # AI: Isolate both artifact existence and result-state discovery: a user's
    # populated workspace can legitimately add arm-level REUSE rows, which must
    # never be mistaken for extra parameter artifacts or Optimizer search work.
    ready_param_paths = set()
    for seed in rolling_robustness.resolved_seeds:
        for arm in rolling_stochastic:
            binding = robustness_plan_runtime._benchmark_param_binding(
                settings=rolling_robust_settings,
                robustness=rolling_robustness,
                arm=arm,
                seed=int(seed),
            )
            ready_param_paths.update((Path(binding["path"]), Path(binding["manifest_path"])))

    _mark_benchmark_step("ready_fixture_setup")
    original_is_file = Path.is_file

    def _isolated_ready_param_is_file(path):
        return path in ready_param_paths or original_is_file(path)

    # A completed result for one arm yields one more REUSE display row than the
    # number of parameter artifacts; the producer actions must remain unchanged.
    result_reuse_arm = rolling_stochastic[0].arm_id
    exact_result_reuse_states = {
        (result_reuse_arm, int(seed)): {
            "action": robustness_plan_runtime.RESULT_ACTION_REUSE,
        }
        for seed in rolling_robustness.resolved_seeds
    }
    isolated_robustness_status = {
        "preparation_plan": SimpleNamespace(actions=[]),
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "artifact_identities": {},
    }
    with (
        patch(
            "services.research.strategy_multi_seed_robustness._discover_registered_strategy_only_scientific_reuse",
            return_value={},
        ),
        patch(
            "services.research.strategy_multi_seed_robustness.inspect_robustness_benchmark_strategy_parameter_artifact",
            return_value={"action": "REUSE", "status": "READY"},
        ),
        patch(
            "services.research.strategy_multi_seed_robustness.resolve_fixed_multiseed_robustness_strategy_parameter_artifact",
            return_value={"action": "REUSE", "status": "READY"},
        ),
        patch(
            "services.research.strategy_multi_seed_robustness.inspect_raw_price_robustness_strategy_parameter_artifact",
            return_value={"action": "REUSE", "status": "READY"},
        ),
        patch.object(Path, "is_file", _isolated_ready_param_is_file),
        patch(
            "services.research.strategy_multi_seed_robustness.compute_strategy_param_scientific_sha256",
            return_value="1" * 64,
        ) as physical_scientific_calls,
        patch(
            "services.research.strategy_multi_seed_robustness.resolve_strategy_param_evaluation_identity_sha256",
            return_value="2" * 64,
        ) as physical_evaluation_calls,
        patch(
            "services.research.strategy_multi_seed_robustness.compute_file_sha256",
            return_value="3" * 64,
        ) as physical_file_sha_calls,
        patch(
            "services.research.strategy_multi_seed_robustness._resolve_existing_scientific_unit_states",
            return_value=({}, {}, exact_result_reuse_states),
        ),
        # AI: This fixture checks *parameter* READY/replay accounting.  The
        # preceding missing-upstream scenario already exercises the real model
        # score inspector against an isolated project root and checks its exact
        # (source, seed) calls.  Do not let this second scenario probe the
        # operator's live Research Rolling artifacts: their contents and size
        # are unrelated to the parameter-plan contract or its time budget.
        patch(
            "services.research.strategy_multi_seed_robustness.inspect_reusable_model_research_rolling_score",
            return_value=(None, "isolated synthetic model-score absence"),
        ),
    ):
        fitted_param_plan, fitted_param_payload = _render_robustness_execution_plan(
            cfg=rolling_robustness,
            settings=rolling_robust_settings,
            status=isolated_robustness_status,
            fixed_arms=rolling_fixed,
            stochastic_arms=rolling_stochastic,
        )
        ready_hash_call_counts = (
            physical_scientific_calls.call_count,
            physical_evaluation_calls.call_count,
            physical_file_sha_calls.call_count,
        )
        # AI: Force an observable artifact revision *during* one parameter
        # snapshot. Hash memoization may not certify mixed-version inputs.
        mutated_revision = {"value": 0}

        def _mutating_parameter_hash(_path):
            mutated_revision["value"] = 1
            return "3" * 64

        with (
            patch.object(
                robustness_plan_runtime,
                "file_change_signature",
                side_effect=lambda _path: ("isolated-revision", mutated_revision["value"]),
            ),
            patch.object(
                robustness_plan_runtime, "compute_file_sha256",
                side_effect=_mutating_parameter_hash,
            ),
        ):
            _, mutation_plan_payload = _render_robustness_execution_plan(
                cfg=rolling_robustness,
                settings=rolling_robust_settings,
                status=isolated_robustness_status,
                fixed_arms=rolling_fixed,
                stochastic_arms=rolling_stochastic,
            )
    ready_param_bindings = fitted_param_payload.get("benchmark_param_bindings") or {}
    ready_result_states = fitted_param_payload.get("robustness_unit_states") or {}
    # A shared physical parameter truth is hashed once, but each arm-seed keeps
    # its own identity row. No long-lived hash cache may bypass later validation.
    physical_views = {
        (str(binding["path"]), str(binding["manifest_path"]), str(binding["evaluation_mode"]))
        for binding in ready_param_bindings.values()
        if not binding.get("result_reuse_only")
    }
    physical_paths = {item[0] for item in physical_views}
    manifest_paths = {item[1] for item in physical_views}
    check_true(
        "robustness_ready_rolling_hashes_shared_parameter_truth_once_per_plan",
        len(ready_param_bindings) >= len(physical_views) > 0
        and ready_hash_call_counts[0] == len(physical_paths)
        and ready_hash_call_counts[1] == len(physical_views)
        and ready_hash_call_counts[2] == len(physical_paths) + len(manifest_paths),
        note=(
            f"arm_seeds={len(ready_param_bindings)} physical_views={len(physical_views)} "
            f"scientific={ready_hash_call_counts[0]} "
            f"evaluation={ready_hash_call_counts[1]} "
            f"file_sha={ready_hash_call_counts[2]}"
        ),
    )
    check_true(
        "robustness_ready_rolling_blocks_artifact_revision_during_hash_snapshot",
        mutated_revision["value"] == 1
        and mutation_plan_payload.get("overall_status") == "BLOCKED"
        and any(
            "benchmark parameter changed during planning" in str(item)
            for item in mutation_plan_payload.get("blockers") or ()
        ),
    )
    check_true(
        "robustness_plan_does_not_count_ready_trained_params_as_pending_optimizer_search",
        fitted_param_payload.get("overall_status") == "READY"
        and not fitted_param_payload.get("blockers")
        and not fitted_param_payload.get("registered_scientific_reuse_units")
        and "0 pending search units" in fitted_param_plan
        and f"scientific result REUSE=0/{expected_total_units} arm×replicate" in fitted_param_plan
        and len(ready_param_bindings) == expected_total_units
        and all(binding.get("plan_action") == "REUSE" for binding in ready_param_bindings.values())
        and len(ready_result_states) == len(rolling_robustness.resolved_seeds)
        and all(
            ready_result_states.get((result_reuse_arm, int(seed)), {}).get("action")
            == robustness_plan_runtime.RESULT_ACTION_REUSE
            for seed in rolling_robustness.resolved_seeds
        ),
    )

    _mark_benchmark_step("ready_rolling_plan")
    return _benchmark_steps, current_robustness
