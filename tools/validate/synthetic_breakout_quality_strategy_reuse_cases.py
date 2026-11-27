from __future__ import annotations

from .synthetic_breakout_quality_support import (
    Path,
    add_check,
    json,
    patch,
    pd,
    tempfile,
    missing_teacher_artifact_fixture,
)


def append_completed_pair_score_reuse_contract_checks(
    *,
    results,
    case_id: str,
    project_root: Path,
) -> None:
    """Validate archived frozen-score provenance independently of profile membership."""

    from core import strategy_compare_policy as strategy_config
    from core.strategy_comparison import (
        STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL,
    )
    from services.research import strategy_comparison as score_reuse_module
    from services.research import strategy_compare_reuse as reuse_owner_module

    reuse_source = (
        project_root / "services" / "research" / "strategy_compare_reuse.py"
    ).read_text(encoding="utf-8")
    orchestration_source = (
        project_root / "services" / "research" / "strategy_comparison.py"
    ).read_text(encoding="utf-8")
    forward_score_reuse_base = strategy_config.get_strategy_comparison_settings(
        "broker_cash_dl_conversion_oos"
    )

    # AI: Presence preflight is not a REUSE decision. A single fresh scandir
    # must agree with the prior per-path is_file check as files change.
    with tempfile.TemporaryDirectory(prefix="strategy_pair_presence_") as temp_dir:
        folder = Path(temp_dir) / "pair"
        required_names = ("strategy_comparison.json", "no_filter_trades.csv", "strategy_comparison.md")
        paths = tuple(folder / name for name in required_names)
        observations = []

        def capture_presence() -> None:
            previous = all(path.is_file() for path in paths)
            current = reuse_owner_module._required_pair_files_present(folder, paths)
            observations.append((previous, current))

        capture_presence()  # no directory
        folder.mkdir()
        capture_presence()  # empty directory
        for name in required_names:
            (folder / name).write_text("fixture\n", encoding="utf-8")
        capture_presence()  # all required files
        (folder / required_names[1]).unlink()
        (folder / required_names[1]).mkdir()
        capture_presence()  # same-name directory is not a regular file
        (folder / required_names[1]).rmdir()
        (folder / required_names[1]).write_text("restored\n", encoding="utf-8")
        capture_presence()  # restored between independent checks
        with patch.object(Path, "is_file", side_effect=AssertionError("unexpected per-file stat")):
            uses_fresh_scandir = reuse_owner_module._required_pair_files_present(folder, paths)
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "required_pair_presence_scandir_matches_per_file_checks_across_mutations",
            True,
            observations == [(False, False), (False, False), (True, True), (False, False), (True, True)]
            and uses_fresh_scandir,
        )

    # AI: Read-side search must not stat dozens of absent historical namespaces,
    # but the ordinary helper still exposes registered prospective paths to
    # callers that create fixture runs. Never cache existence across searches.
    with tempfile.TemporaryDirectory(prefix="strategy_compare_cache_roots_") as temp_dir:
        discovery_root = Path(temp_dir)
        declared_namespaces = strategy_config.resolve_strategy_comparison_result_search_roots(
            forward_score_reuse_base
        )
        expected_first = discovery_root / declared_namespaces[0] / "runs"
        prospective = reuse_owner_module._comparison_runs_roots(
            root=discovery_root, settings=forward_score_reuse_base
        )
        missing = reuse_owner_module._comparison_runs_roots(
            root=discovery_root, settings=forward_score_reuse_base,
            existing_only=True,
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "result_root_discovery_preserves_prospective_registered_namespaces",
            True,
            len(prospective) == len(declared_namespaces)
            and prospective[0] == expected_first
            and missing == (),
        )
        expected_first.mkdir(parents=True)
        visible = reuse_owner_module._comparison_runs_roots(
            root=discovery_root, settings=forward_score_reuse_base,
            existing_only=True,
        )
        expected_first.rmdir()
        removed = reuse_owner_module._comparison_runs_roots(
            root=discovery_root, settings=forward_score_reuse_base,
            existing_only=True,
        )
        expected_first.mkdir()
        recreated = reuse_owner_module._comparison_runs_roots(
            root=discovery_root, settings=forward_score_reuse_base,
            existing_only=True,
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "result_root_read_side_discovery_rechecks_publish_and_removal",
            True,
            visible == (expected_first.resolve(),)
            and removed == ()
            and recreated == (expected_first.resolve(),),
        )
        with patch.object(
            reuse_owner_module,
            "resolve_strategy_comparison_result_search_roots",
            return_value=("../invalid",),
        ):
            try:
                reuse_owner_module._comparison_runs_roots(
                    root=discovery_root, settings=forward_score_reuse_base,
                    existing_only=True,
                )
            except ValueError:
                invalid_rejected = True
            else:
                invalid_rejected = False
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "result_root_existing_only_never_bypasses_path_validation",
            True, invalid_rejected,
        )

    # A completed result is only a reusable candidate while its scientific Params
    # producer is still BUILD/REBUILD/RESUME.  The final REUSE/RUN decision must be
    # made after canonical parameter production completes and the plan is re-resolved.
    parent_child_settings = strategy_config.get_strategy_comparison_settings(
        "non_dl_broker_cash_oos"
    )
    parent_param_sources = tuple(
        dict.fromkeys(arm.param_source for arm in parent_child_settings.enabled_arms)
    )
    pending_plan = reuse_owner_module.ResearchArtifactPlan.from_actions(
        reuse_owner_module.ResearchArtifactAction(
            action_id=f"param:{source_id}",
            artifact_key=f"param:{source_id}",
            action="REBUILD",
            builder_type="synthetic_param_builder",
            description="synthetic pending parameter producer",
            path=f"models/strategy_params/{source_id}.json",
            dependencies=(),
            producer_work_type="optimizer_strategy_parameter_service",
        )
        for source_id in parent_param_sources
    )
    ready_plan = reuse_owner_module.ResearchArtifactPlan.from_actions(
        reuse_owner_module.ResearchArtifactAction(
            action_id=f"param:{source_id}",
            artifact_key=f"param:{source_id}",
            action="REUSE",
            builder_type=None,
            description="synthetic ready parameter producer",
            path=f"models/strategy_params/{source_id}.json",
            dependencies=(),
            producer_work_type="existing_artifact",
        )
        for source_id in parent_param_sources
    )
    replay_status_base = {
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
    }
    synthetic_completed_pair = Path("synthetic_completed_pair").resolve()
    with patch.object(
        reuse_owner_module,
        "_find_reusable_baseline_source",
        return_value=synthetic_completed_pair,
    ):
        pending_cache = reuse_owner_module._collect_replay_cache_status(
            root=project_root,
            settings=parent_child_settings,
            status={**replay_status_base, "preparation_plan": pending_plan},
        )
        ready_cache = reuse_owner_module._collect_replay_cache_status(
            root=project_root,
            settings=parent_child_settings,
            status={**replay_status_base, "preparation_plan": ready_plan},
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_arm_reuse_waits_for_pending_parameter_parent_before_final_reuse",
        True,
        all(
            dict(pending_cache["arm_states"][arm.arm_id]).get("action") == "WAIT"
            and bool(
                dict(pending_cache["arm_states"][arm.arm_id]).get(
                    "scientific_result_ready"
                )
            )
            and dict(ready_cache["arm_states"][arm.arm_id]).get("action") == "REUSE"
            for arm in parent_child_settings.enabled_arms
        ),
    )

    # Regression: adding DL arms to a previously completed No-DL suite must
    # not make its shared C96 scientific observation appear as a new RUN.  The
    # baseline search must examine paired as well as standalone DL-off arms.
    broker_settings = strategy_config.get_strategy_comparison_settings(
        "broker_cash_dl_conversion_rolling"
    )
    broker_plan = reuse_owner_module.ResearchArtifactPlan.from_actions(
        reuse_owner_module.ResearchArtifactAction(
            action_id=f"param:{source}",
            artifact_key=f"param:{source}",
            action="REUSE",
            builder_type=None,
            description="synthetic identity-compatible parameters",
            path=f"models/strategy_params/{source}.json",
            producer_work_type="existing_artifact",
        )
        for source in dict.fromkeys(
            arm.param_source for arm in broker_settings.enabled_arms
        )
    )
    c96_baseline = project_root / "synthetic_c96_completed_baseline"

    def find_broker_baseline(**kwargs):
        arm = kwargs["off_arm"]
        return c96_baseline if arm.arm_id == "C96" else None

    with patch.object(
        reuse_owner_module, "_find_reusable_pair", return_value=None
    ), patch.object(
        reuse_owner_module,
        "_find_reusable_baseline_source",
        side_effect=find_broker_baseline,
    ) as baseline_lookup:
        broker_cache = reuse_owner_module._collect_replay_cache_status(
            root=project_root,
            settings=broker_settings,
            status={
                **replay_status_base,
                "preparation_plan": broker_plan,
            },
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "paired_c96_reference_reuses_verified_old_baseline_without_dl_pair_cache",
        True,
        (
            broker_cache["arm_states"]["C96"]["action"] == "REUSE"
            and broker_cache["arm_states"]["C96"]["source_dir"]
            == c96_baseline.resolve()
            and all(
                broker_cache["arm_states"][arm.arm_id]["action"] == "RUN"
                for arm in broker_settings.enabled_arms
                if arm.dl_enabled
            )
            and any(
                call.kwargs["off_arm"].arm_id == "C96"
                for call in baseline_lookup.call_args_list
            )
        ),
    )

    # Execution plan has only one aggregate reporting step.  Contrast metadata
    # remains enabled for the unchanged canonical result/report contract.
    with patch.object(
        score_reuse_module,
        "render_strategy_execution_plan_surface",
        side_effect=lambda **kwargs: kwargs["action_rows"],
    ):
        broker_plan_rows = score_reuse_module.render_execution_plan(
            settings=broker_settings,
            status={
                "preparation_plan": broker_plan,
                "replay_cache": broker_cache,
                "config_fingerprint": "synthetic-broker-cash",
            },
        )
    report_steps = [row for row in broker_plan_rows if row[0] == "REPORT"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "compare_execution_plan_has_one_aggregate_report_not_pairwise_report_jobs",
        True,
        (
            len(report_steps) == 1
            and "綜合報表" in str(report_steps[0][1])
            and len(broker_settings.enabled_contrasts) > 1
            and all(
                str(contrast.contrast_id)
                not in str(report_steps[0][1])
                for contrast in broker_settings.enabled_contrasts
            )
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_cache_waives_source_only_when_no_arm_needs_runtime_source",
        True,
        all(
            token in reuse_source
            for token in (
                "archived_completed_pair",
                "def _find_reusable_pair_with_archived_source(",
                "def _apply_completed_pair_dependency_waivers(",
                "all(isinstance(pairs.get(arm.arm_id), dict) for arm in dependent_arms)",
                "本次使用此DL的arms全部重用identity一致的completed pair",
            )
        )
        and "def _find_reusable_pair_with_archived_source(" not in orchestration_source
        and "def _apply_completed_pair_dependency_waivers(" not in orchestration_source,
    )
    # Selection-PIT cache identity must track only score columns consumed by each arm.
    selection_settings = strategy_config.get_strategy_comparison_settings(
        "broker_cash_dl_conversion_oos"
    )
    selection_period = {"start": "2021-01-01", "end": "2026-03-02"}
    projection_artifacts = {
        "dl:CONT13K_ROLL:manifest": {"sha256": "manifest-v1"},
        "dl:CONT13K_ROLL:audit": {"sha256": "audit-v1"},
        "dl:CONT13K_ROLL:forward_scores": {
            "sha256": "whole-file-v1",
            "score_projection_sha256": {
                "breakout_quality_score": "primary-v1",
                "raw_safety_score": "safety-v1",
            },
        },
    }
    auxiliary_changed = json.loads(json.dumps(projection_artifacts))
    auxiliary_changed["dl:CONT13K_ROLL:manifest"]["sha256"] = "manifest-v2"
    auxiliary_changed["dl:CONT13K_ROLL:audit"]["sha256"] = "audit-v2"
    auxiliary_changed["dl:CONT13K_ROLL:forward_scores"]["sha256"] = "whole-file-v2"
    auxiliary_changed["dl:CONT13K_ROLL:forward_scores"]["score_projection_sha256"][
        "raw_safety_score"
    ] = "safety-v2"
    primary_changed = json.loads(json.dumps(auxiliary_changed))
    primary_changed["dl:CONT13K_ROLL:forward_scores"]["score_projection_sha256"][
        "breakout_quality_score"
    ] = "primary-v2"

    def projected_fp(arm_id, artifacts):
        return score_reuse_module._pair_cache_fingerprint_from_payload(
            settings_payload=selection_settings.as_dict(),
            artifact_identities=artifacts,
            comparison_period=selection_period,
            off_arm_payload=selection_settings.arms["C96"].as_dict(),
            on_arm_payload=selection_settings.arms[arm_id].as_dict(),
            engine_schema_version=score_reuse_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
            parameter_evaluation_sha256="param-sha",
        )

    exact_base = projected_fp("C111", projection_artifacts)
    exact_aux = projected_fp("C111", auxiliary_changed)
    exact_primary = projected_fp("C111", primary_changed)
    direct_base = projected_fp("C112", projection_artifacts)
    direct_aux = projected_fp("C112", auxiliary_changed)
    direct_primary = projected_fp("C112", primary_changed)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_pit_pair_identity_ignores_unconsumed_auxiliary_score_but_tracks_consumed_columns",
        True,
        exact_base == exact_aux
        and direct_base == direct_aux
        and exact_base != exact_primary
        and direct_base != direct_primary,
    )

    gate_artifacts = {
        "dl:CONT13CO_ROLL:forward_scores": {
            "sha256": "rank-file-v1", "score_projection_sha256": {"breakout_quality_score": "rank-v1"},
        },
        "dl:CONT13CP_ROLL:forward_scores": {
            "sha256": "positive-file-v1", "score_projection_sha256": {"positive_p_positive": "positive-v1"},
        },
    }
    changed_gate_artifacts = json.loads(json.dumps(gate_artifacts))
    changed_gate_artifacts["dl:CONT13CP_ROLL:forward_scores"]["score_projection_sha256"]["positive_p_positive"] = "positive-v2"
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "selection_pit_pair_identity_tracks_current_consumed_positive_companion_only",
        True,
        projected_fp("C123", gate_artifacts) == projected_fp("C123", changed_gate_artifacts)
        and projected_fp("C125", gate_artifacts) != projected_fp("C125", changed_gate_artifacts),
    )

    from filters.breakout_quality.strategy_score_projection import (
        compute_score_projection_sha256s,
    )
    with tempfile.TemporaryDirectory() as projection_tmp:
        projection_root = Path(projection_tmp).resolve()
        score_path = projection_root / "models" / "pure_mfe_scores.csv"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        score_frame = pd.DataFrame(
            [
                {
                    "ticker": "2330",
                    "date": "2021-01-04",
                    "group_index": 1,
                    "breakout_quality_score": 0.90,
                    "raw_safety_score": 0.80,
                    "fold_id": "fold_20210101_20260302",
                    "model_information_cutoff": "2020-12-31",
                },
                {
                    "ticker": "2317",
                    "date": "2021-01-05",
                    "group_index": 2,
                    "breakout_quality_score": 0.70,
                    "raw_safety_score": 0.60,
                    "fold_id": "fold_20210101_20260302",
                    "model_information_cutoff": "2020-12-31",
                },
            ]
        )
        score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")
        projection_map = compute_score_projection_sha256s(
            score_path=score_path,
            score_source="selection_point_in_time",
            columns=("breakout_quality_score", "raw_safety_score"),
        )
        on_arm = selection_settings.arms["C111"]
        off_arm = selection_settings.arms["C96"]
        group_id = score_reuse_module._pair_group_id(
            param_source=on_arm.param_source,
            rule_policy=on_arm.rule_policy,
            dl_id=str(on_arm.dl_id or ""),
            dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
        )
        run_dir = (
            projection_root
            / selection_settings.output_root
            / "runs"
            / "20260826_000000_legacy_projection"
        )
        pair_dir = run_dir / "pairs" / group_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        required_paths = score_reuse_module._pair_cache_required_files(
            pair_dir, on_arm=on_arm
        )
        for required_path in required_paths:
            required_path.parent.mkdir(parents=True, exist_ok=True)
            if required_path.name == "score_ranking_orderable_candidates.csv":
                pd.DataFrame(
                    [
                        {
                            "ticker": "2330",
                            "signal_date": "2021-01-04",
                            "breakout_quality_score": 0.90,
                            "breakout_quality_score_date": "2021-01-04",
                            "breakout_quality_score_available": True,
                        },
                        {
                            "ticker": "2317",
                            "signal_date": "2021-01-05",
                            "breakout_quality_score": 0.70,
                            "breakout_quality_score_date": "2021-01-05",
                            "breakout_quality_score_available": True,
                        },
                    ]
                ).to_csv(required_path, index=False, encoding="utf-8-sig")
            elif required_path.suffix == ".json":
                required_path.write_text("{}\n", encoding="utf-8")
            else:
                required_path.write_text("x\n", encoding="utf-8")
        legacy_artifacts = {
            "dl:CONT13K_ROLL:manifest": {"path": "models/old_manifest.json", "sha256": "old-manifest"},
            "dl:CONT13K_ROLL:audit": {"path": "models/old_audit.json", "sha256": "old-audit"},
            "dl:CONT13K_ROLL:forward_scores": {"path": "models/pure_mfe_scores.csv", "sha256": "old-whole-file"},
        }
        pair_payload = {
            "metadata": {
                "schema_version": score_reuse_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
                "params_file_sha256": "param-sha",
                "comparison_period": selection_period,
            }
        }
        # A realistic completed run publishes the identical pair payload both
        # locally and in the aggregated result.  Missing/stale local JSON is not
        # valid REUSE evidence even when all filenames are present.
        (pair_dir / "strategy_comparison.json").write_text(
            json.dumps(pair_payload, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "status": "COMPLETED",
                    "settings": selection_settings.as_dict(),
                    "artifact_identities": legacy_artifacts,
                    "comparison_period": selection_period,
                    "pairs": {group_id: pair_payload},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        current_artifacts = {
            "dl:CONT13K_ROLL:manifest": {"path": "models/new_manifest.json", "sha256": "new-manifest"},
            "dl:CONT13K_ROLL:audit": {"path": "models/new_audit.json", "sha256": "new-audit"},
            "dl:CONT13K_ROLL:forward_scores": {
                "path": "models/pure_mfe_scores.csv",
                "sha256": "new-whole-file",
                "score_projection_sha256": projection_map,
            },
        }
        status = {
            "artifact_identities": current_artifacts,
            "comparison_period": selection_period,
            "resolved_arm_parameter_identities": {"C111": {"sha256": "param-sha"}},
        }
        migrated_hit = score_reuse_module._find_reusable_pair(
            root=projection_root,
            settings=selection_settings,
            status=status,
            off_arm=off_arm,
            on_arm=on_arm,
        )
        # Verify the actually registered cache lookup, including legacy PIT
        # projection migration, rejects a divergent pair-local result.
        local_json = pair_dir / "strategy_comparison.json"
        local_json.write_text("not JSON", encoding="utf-8")
        corrupt_current_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        divergent_pair = json.loads(json.dumps(pair_payload))
        divergent_pair["metadata"]["comparison_period"]["end"] = "2099-12-31"
        local_json.write_text(json.dumps(divergent_pair), encoding="utf-8")
        identity_conflict_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        divergent_pair = json.loads(json.dumps(pair_payload))
        divergent_pair["no_filter"] = {"tampered": True}
        local_json.write_text(json.dumps(divergent_pair), encoding="utf-8")
        metric_conflict_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        local_json.write_text(
            json.dumps(pair_payload, ensure_ascii=False, indent=3), encoding="utf-8"
        )
        formatting_only_cache_hit = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is not None
        archived_status = {
            "artifact_identities": {},
            "comparison_period": selection_period,
            "dl_sources": {str(on_arm.dl_id): {"ready": False}},
            "resolved_arm_parameter_identities": {
                on_arm.arm_id: {"sha256": "param-sha"}
            },
        }
        archived_hit = score_reuse_module._find_reusable_pair_with_archived_source(
            root=projection_root, settings=selection_settings,
            status=archived_status, off_arm=off_arm, on_arm=on_arm,
        )
        (pair_dir / "strategy_comparison.json").write_text(
            "not JSON", encoding="utf-8"
        )
        archived_corrupt_rejected = (
            score_reuse_module._find_reusable_pair_with_archived_source(
                root=projection_root, settings=selection_settings,
                status=archived_status, off_arm=off_arm, on_arm=on_arm,
            ) is None
        )
        (pair_dir / "strategy_comparison.json").write_text(
            json.dumps(pair_payload, ensure_ascii=False), encoding="utf-8"
        )

        # A new completed run pins the final CSV/Markdown bytes, independent
        # of the scientific pair-cache fingerprint. Test the REAL normal and
        # archived lookup paths, not only the digest helper in isolation.
        from services.research.strategy_compare_pair_integrity import (
            PAIR_FILE_INTEGRITY_FIELD,
            build_completed_pair_file_integrity,
            completed_pair_file_integrity_matches,
        )
        auxiliary_json = pair_dir / "post_replay_diagnostics.json"
        auxiliary_json.write_text('{"diagnostics": 1}\n', encoding="utf-8")
        pinned_integrity = build_completed_pair_file_integrity({group_id: pair_dir})
        report_path = run_dir / "strategy_comparison.json"
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload[PAIR_FILE_INTEGRITY_FIELD] = pinned_integrity
        report_path.write_text(json.dumps(report_payload), encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({PAIR_FILE_INTEGRITY_FIELD: pinned_integrity}), encoding="utf-8"
        )
        required_pair_files = score_reuse_module._pair_cache_required_files(
            pair_dir, on_arm=on_arm
        )
        pinned_current_hit = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        )
        pinned_archived_hit = score_reuse_module._find_reusable_pair_with_archived_source(
            root=projection_root, settings=selection_settings,
            status=archived_status, off_arm=off_arm, on_arm=on_arm,
        )
        csv_path = pair_dir / "no_filter_trades.csv"
        saved_csv = csv_path.read_bytes()
        csv_path.write_bytes(saved_csv + b"\nmodified\n")
        changed_csv_current_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        changed_csv_archived_rejected = score_reuse_module._find_reusable_pair_with_archived_source(
            root=projection_root, settings=selection_settings,
            status=archived_status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        csv_path.write_bytes(saved_csv)

        markdown_path = pair_dir / "strategy_comparison.md"
        saved_markdown = markdown_path.read_bytes()
        markdown_path.write_bytes(saved_markdown + b"\nmodified\n")
        changed_markdown_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        markdown_path.write_bytes(saved_markdown)
        auxiliary_json.write_text('{"diagnostics": 2}\n', encoding="utf-8")
        changed_auxiliary_json_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        auxiliary_json.write_text('{"diagnostics": 1}\n', encoding="utf-8")

        manifest_path.unlink()
        missing_manifest_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        archived_missing_manifest_rejected = score_reuse_module._find_reusable_pair_with_archived_source(
            root=projection_root, settings=selection_settings,
            status=archived_status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        manifest_path.write_text(
            json.dumps({PAIR_FILE_INTEGRITY_FIELD: pinned_integrity}), encoding="utf-8"
        )
        truncated_integrity = json.loads(json.dumps(pinned_integrity))
        del truncated_integrity["pairs"][group_id]["no_filter_trades.csv"]
        manifest_path.write_text(
            json.dumps({PAIR_FILE_INTEGRITY_FIELD: truncated_integrity}), encoding="utf-8"
        )
        truncated_manifest_rejected = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is None
        manifest_path.write_text(
            json.dumps({PAIR_FILE_INTEGRITY_FIELD: pinned_integrity}), encoding="utf-8"
        )

        # Formatting the already semantically verified pair JSON is intentionally
        # harmless; its content equality was protected in the previous SSOT batch.
        local_json = pair_dir / "strategy_comparison.json"
        local_json.write_text(json.dumps(pair_payload, indent=4), encoding="utf-8")
        formatted_pair_json_still_reusable = score_reuse_module._find_reusable_pair(
            root=projection_root, settings=selection_settings,
            status=status, off_arm=off_arm, on_arm=on_arm,
        ) is not None
        pinned_low_level_valid = completed_pair_file_integrity_matches(
            run_dir=run_dir, run_payload=report_payload,
            pair_id=group_id, pair_dir=pair_dir,
            required_paths=required_pair_files,
        )
        # AI: Completed-pair inventory must use fresh directory-entry metadata.
        # Rechecking each already-listed file with Path.is_file() is avoidable
        # Windows I/O, but symlinks must still fail closed.
        from contextlib import nullcontext
        from services.research import strategy_compare_pair_integrity as pair_integrity_owner
        with patch.object(
            Path, "is_file", side_effect=AssertionError("redundant pair-file stat")
        ):
            inventory = pair_integrity_owner._tracked_pair_files(pair_dir)
        inventory_uses_dir_entries = (
            "strategy_comparison.md" in inventory
            and "no_filter_trades.csv" in inventory
            and "post_replay_diagnostics.json" in inventory
            and "strategy_comparison.json" not in inventory
        )

        newly_published = pair_dir / "new_post_replay_auxiliary.csv"
        newly_published.write_text("fresh\n", encoding="utf-8")
        published_discovered = newly_published.name in pair_integrity_owner._tracked_pair_files(pair_dir)
        newly_published.unlink()
        removed_disappears = newly_published.name not in pair_integrity_owner._tracked_pair_files(pair_dir)
        non_file_path = pair_dir / "directory_named_csv.csv"
        non_file_path.mkdir()
        directory_not_tracked = non_file_path.name not in pair_integrity_owner._tracked_pair_files(pair_dir)
        non_file_path.rmdir()

        class _SymlinkEntry:
            name = "unsafe.csv"

            def is_symlink(self):
                return True

            def is_file(self):
                raise AssertionError("symlink type must not be followed")

        with patch.object(
            pair_integrity_owner.os, "scandir",
            return_value=nullcontext((_SymlinkEntry(),)),
        ):
            try:
                pair_integrity_owner._tracked_pair_files(pair_dir)
            except ValueError as exc:
                symlink_inventory_rejected = "符號連結" in str(exc)
            else:
                symlink_inventory_rejected = False
        score_frame.loc[0, "breakout_quality_score"] = 0.91
        score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")
        changed_projection_map = compute_score_projection_sha256s(
            score_path=score_path,
            score_source="selection_point_in_time",
            columns=("breakout_quality_score", "raw_safety_score"),
        )
        changed_current_artifacts = json.loads(json.dumps(current_artifacts))
        changed_current_artifacts["dl:CONT13K_ROLL:forward_scores"][
            "score_projection_sha256"
        ] = changed_projection_map
        migrated_miss = score_reuse_module._find_reusable_pair(
            root=projection_root,
            settings=selection_settings,
            status={
                **status,
                "artifact_identities": changed_current_artifacts,
            },
            off_arm=off_arm,
            on_arm=on_arm,
        )
        # Regression: same mtime AND size, different bytes. The physical file
        # SHA must supersede the old metadata-only projection-cache key.
        import os
        from core.file_integrity import compute_file_sha256
        from filters.breakout_quality import ranking_score_store as score_owner
        from filters.breakout_quality import strategy_score_projection as projection_owner
        cached_table = score_owner.load_selection_point_in_time_score_table_from_path(str(score_path))
        stamp = score_path.stat()
        old_bytes = score_path.read_bytes()
        assert b"0.91" in old_bytes
        score_path.write_bytes(old_bytes.replace(b"0.91", b"0.92", 1))
        os.utime(score_path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        same_stat_projection = compute_score_projection_sha256s(
            score_path=score_path,
            score_source="selection_point_in_time",
            columns=("breakout_quality_score", "raw_safety_score"),
            score_file_sha256=compute_file_sha256(score_path),
        )
        projection_was_cached = projection_owner._selection_pit_projection_cached.cache_info().currsize > 0
        score_owner.clear_selection_point_in_time_ranking_contract_cache()
        dependent_projection_was_invalidated = (
            projection_was_cached
            and projection_owner._selection_pit_projection_cached.cache_info().currsize == 0
        )
        refreshed_table = score_owner.load_selection_point_in_time_score_table_from_path(str(score_path))
        same_stat_collision_blocked = (
            (score_path.stat().st_size, score_path.stat().st_mtime_ns)
            == (stamp.st_size, stamp.st_mtime_ns)
            and same_stat_projection["breakout_quality_score"]
            != changed_projection_map["breakout_quality_score"]
            and same_stat_projection["raw_safety_score"]
            == changed_projection_map["raw_safety_score"]
            and float(cached_table.loc[("2330", "2021-01-04"), "breakout_quality_score"]) == 0.91
            and float(refreshed_table.loc[("2330", "2021-01-04"), "breakout_quality_score"]) == 0.92
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_cache_rejects_corrupt_identity_or_payload_but_not_json_formatting",
        True,
        isinstance(migrated_hit, dict)
        and corrupt_current_rejected
        and identity_conflict_rejected
        and metric_conflict_rejected
        and formatting_only_cache_hit,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "archived_completed_pair_reuse_requires_readable_matching_local_json",
        True,
        isinstance(archived_hit, dict) and archived_corrupt_rejected,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_integrity_pins_csv_and_markdown_for_normal_and_archived_reuse",
        True,
        isinstance(pinned_current_hit, dict)
        and isinstance(pinned_archived_hit, dict)
        and pinned_low_level_valid
        and changed_csv_current_rejected
        and changed_csv_archived_rejected
        and changed_markdown_rejected
        and changed_auxiliary_json_rejected,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "completed_pair_inventory_uses_fresh_scandir_metadata_without_extra_stat_and_rejects_symlink",
        True, inventory_uses_dir_entries and symlink_inventory_rejected
        and published_discovered and removed_disappears and directory_not_tracked,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_integrity_rejects_missing_or_truncated_manifest_but_allows_pair_json_reformatting",
        True,
        missing_manifest_rejected
        and archived_missing_manifest_rejected
        and truncated_manifest_rejected
        and formatted_pair_json_still_reusable,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_projection_and_pit_cache_invalidate_same_stat_changed_bytes",
        True, same_stat_collision_blocked,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "pit_score_owner_invalidates_registered_projection_cache_without_reverse_import",
        True, dependent_projection_was_invalidated,
    )
    # Regression: a published Forward OOS table may be atomically replaced at
    # the same path during a multi-seed replay. A cached score/contract must not
    # be reused merely because the path and the restored mtime/size match.
    from filters.breakout_quality import ranking_score_store as forward_owner
    from types import SimpleNamespace
    import os
    with tempfile.TemporaryDirectory(prefix="forward_score_revision_") as temp_dir:
        forward_path = Path(temp_dir) / "forward.csv"
        pit_path = Path(temp_dir) / "pit.csv"
        profile = "daily_universal_shared_safety_hs_conditional_mfe_full_list_ndcg_pairwise"
        forward_frame = pd.DataFrame({
            "ticker": ["2330"], "date": ["2021-01-04"],
            "group_index": [0], "model_score": [0.20],
            # Shared-safety multi-output profiles require this sidecar even
            # when the test is about score-file replacement, not safety.
            "raw_safety_score": [0.60],
        })
        # Reject an incomplete publication before exercising same-path reuse.
        # Otherwise the revision-cache test never reaches its actual subject.
        forward_frame.drop(columns="raw_safety_score").to_csv(
            forward_path, index=False, encoding="utf-8-sig"
        )
        try:
            forward_owner.load_continuous_ranker_oos_score_table_from_path(
                str(forward_path), profile
            )
        except ValueError as exc:
            missing_sidecar_rejected = "raw_safety_score" in str(exc)
        else:
            missing_sidecar_rejected = False
        forward_frame.to_csv(forward_path, index=False, encoding="utf-8-sig")
        old_forward = forward_owner.load_continuous_ranker_oos_score_table_from_path(
            str(forward_path), profile
        )
        with patch.object(forward_owner, "resolve_continuous_ranker_oos_score_path", return_value=forward_path):
            old_canonical = forward_owner.load_continuous_ranker_oos_score_table(
                temp_dir, "fixture", "fixture", profile
            )
            stamp = forward_path.stat()
            staged = forward_path.with_suffix(".staged")
            forward_frame["model_score"] = 0.90
            forward_frame.to_csv(staged, index=False, encoding="utf-8-sig")
            os.utime(staged, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            os.replace(staged, forward_path)
            new_forward = forward_owner.load_continuous_ranker_oos_score_table_from_path(
                str(forward_path), profile
            )
            new_canonical = forward_owner.load_continuous_ranker_oos_score_table(
                temp_dir, "fixture", "fixture", profile
            )
            old_value = float(old_forward.loc[("2330", "2021-01-04"), "model_score"])
            changed_value = float(new_forward.loc[("2330", "2021-01-04"), "model_score"])
            canonical_value = float(new_canonical.loc[("2330", "2021-01-04"), "model_score"])
            signature_collision = (
                forward_path.stat().st_size == stamp.st_size
                and forward_path.stat().st_mtime_ns == stamp.st_mtime_ns
            )
            # Fail closed when a revision violates the existing score domain;
            # a formerly valid cached DataFrame is not acceptable evidence.
            forward_frame["model_score"] = -0.90
            forward_frame.to_csv(staged, index=False, encoding="utf-8-sig")
            os.replace(staged, forward_path)
            try:
                forward_owner.load_continuous_ranker_oos_score_table_from_path(
                    str(forward_path), profile
                )
            except ValueError:
                invalid_forward_rejected = True
            else:
                invalid_forward_rejected = False

        # The same physical revision hint must cover isolated PIT path overrides
        # that have no manifest/hash input from a Strategy Compare planner.
        pit_frame = pd.DataFrame({
            "ticker": ["2330"], "date": ["2021-01-04"],
            "group_index": [0], "breakout_quality_score": [0.25],
            "raw_safety_score": [0.60],
            "fold_id": ["fold_fixture"],
            "model_information_cutoff": ["2020-12-31"],
        })
        pit_frame.drop(columns="raw_safety_score").to_csv(
            pit_path, index=False, encoding="utf-8-sig"
        )
        try:
            forward_owner.load_selection_point_in_time_score_table_from_path(
                str(pit_path), experiment_profile=profile
            )
        except ValueError as exc:
            missing_pit_sidecar_rejected = "raw_safety_score" in str(exc)
        else:
            missing_pit_sidecar_rejected = False
        pit_frame.to_csv(pit_path, index=False, encoding="utf-8-sig")
        old_pit = forward_owner.load_selection_point_in_time_score_table_from_path(
            str(pit_path), experiment_profile=profile
        )
        pit_stamp = pit_path.stat()
        pit_staged = pit_path.with_suffix(".staged")
        pit_frame["breakout_quality_score"] = 0.85
        pit_frame.to_csv(pit_staged, index=False, encoding="utf-8-sig")
        os.utime(pit_staged, ns=(pit_stamp.st_atime_ns, pit_stamp.st_mtime_ns))
        os.replace(pit_staged, pit_path)
        new_pit = forward_owner.load_selection_point_in_time_score_table_from_path(
            str(pit_path), experiment_profile=profile
        )
        pit_reloaded = (
            pit_path.stat().st_size == pit_stamp.st_size
            and pit_path.stat().st_mtime_ns == pit_stamp.st_mtime_ns
            and float(old_pit.loc[("2330", "2021-01-04"), "breakout_quality_score"]) == 0.25
            and float(new_pit.loc[("2330", "2021-01-04"), "breakout_quality_score"]) == 0.85
        )

        # A changed canonical manifest/report/score publication must cause the
        # canonical FIT/score contract to revalidate, not return the old object.
        manifest = Path(temp_dir) / "manifest.json"
        model = Path(temp_dir) / "model.pt"
        report = Path(temp_dir) / "continuous_ranker_report.json"
        for item in (manifest, model, report):
            item.write_bytes(b"synthetic contract identity")
        artifacts = SimpleNamespace(manifest_path=manifest, model_path=model)
        forward_owner.load_continuous_ranker_oos_contract.cache_clear()
        with patch.object(forward_owner, "resolve_filter_artifact_paths", return_value=artifacts), patch.object(
            forward_owner, "resolve_filter_model_output_dir", return_value=Path(temp_dir)
        ), patch.object(
            forward_owner, "resolve_continuous_ranker_oos_score_path", return_value=forward_path
        ), patch.object(
            forward_owner, "_load_continuous_ranker_oos_contract_uncached", side_effect=lambda *a, **k: object()
        ) as validation:
            once = forward_owner.load_continuous_ranker_oos_contract(temp_dir, "fixture", "fixture", profile)
            cached = forward_owner.load_continuous_ranker_oos_contract(temp_dir, "fixture", "fixture", profile)
            manifest.write_bytes(b"new contract identity")
            refreshed = forward_owner.load_continuous_ranker_oos_contract(temp_dir, "fixture", "fixture", profile)
            contract_rerun = once is cached and refreshed is not cached and validation.call_count == 2
        forward_owner.load_continuous_ranker_oos_contract.cache_clear()

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_oos_same_path_republication_invalidates_override_and_canonical_score_cache",
        True, old_value == 0.20 and changed_value == 0.90 and canonical_value == 0.90
        and float(old_canonical.loc[("2330", "2021-01-04"), "model_score"]) == 0.20
        and signature_collision and invalid_forward_rejected
        and missing_sidecar_rejected,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "pit_isolated_same_stat_atomic_replacement_reloads_without_manual_cache_clear",
        True, pit_reloaded and missing_pit_sidecar_rejected,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "forward_canonical_contract_revalidates_changed_publication",
        True, contract_rerun,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "legacy_selection_pit_pair_reuses_after_auxiliary_column_migration_only_when_recorded_primary_scores_match",
        True,
        isinstance(migrated_hit, dict)
        and migrated_hit.get("source_artifact_mode")
        == "legacy_pair_runtime_score_projection_verified"
        and migrated_miss is None,
    )


def append_completed_pair_baseline_integrity_contract_checks(
    *, results, case_id: str
) -> None:
    """Exercise the shared-baseline cache selector with pinned and legacy runs."""

    from core.strategy_compare_policy import get_strategy_comparison_settings
    from core.strategy_comparison import (
        resolve_strategy_comparison_arm_param_policy,
        resolve_strategy_parameter_consumption_mode,
    )
    from services.research import strategy_compare_reuse as reuse
    from services.research.strategy_compare_pair_integrity import (
        PAIR_FILE_INTEGRITY_FIELD,
        build_completed_pair_file_integrity,
    )
    from filters.breakout_quality.strategy_compare_sources import (
        OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        OPTIONAL_ENTRY_FILTER_POLICY_CURRENT,
    )
    from filters.breakout_quality.strategy_rule_policies import ALL_RULE_FILTERS_OFF_OVERRIDES

    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_oos")
    off_arm = next(arm for arm in settings.arms.values() if not arm.dl_enabled)
    period = {"start": "2021-01-01", "end": "2022-12-31"}
    status = {
        "comparison_period": period,
        "resolved_arm_parameter_identities": {
            off_arm.arm_id: {"sha256": "synthetic-param-sha"}
        },
    }
    all_off = off_arm.rule_policy == "all_off"
    metadata = {
        "dataset": settings.dataset,
        "params_file_sha256": "synthetic-param-sha",
        "requested_param_policy": resolve_strategy_comparison_arm_param_policy(
            settings, off_arm
        ),
        "param_evaluation_mode": resolve_strategy_parameter_consumption_mode(
            settings.parameter_sources[off_arm.param_source]
        ),
        "optional_entry_filter_policy": (
            OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF
            if all_off else OPTIONAL_ENTRY_FILTER_POLICY_CURRENT
        ),
        "shared_param_overrides": (
            dict(ALL_RULE_FILTERS_OFF_OVERRIDES) if all_off else {}
        ),
        "max_positions": int(settings.max_positions),
        "enable_rotation": settings.rotation == "on",
        "comparison_period": period,
    }
    required_names = (
        "strategy_comparison.json", "strategy_comparison.md",
        "yearly_returns_comparison.csv", "no_filter_equity.csv",
        "no_filter_trades.csv", "no_filter_daily_capacity.csv",
        "no_filter_orderable_candidates.csv",
    )
    with tempfile.TemporaryDirectory(prefix="ssot_pair_baseline_") as tmp:
        root = Path(tmp)
        runs_root = reuse._comparison_runs_roots(root=root, settings=settings)[0]
        run_dir = runs_root / "20990101_000000_integrity_synthetic"
        pair_id = "baseline_synthetic"
        pair_dir = run_dir / "pairs" / pair_id
        pair_dir.mkdir(parents=True)
        pair_payload = {"metadata": metadata, "no_filter": {"total_return_pct": 1.0}}
        for name in required_names:
            (pair_dir / name).write_text(
                json.dumps(pair_payload) if name.endswith(".json") else "baseline\n",
                encoding="utf-8",
            )
        run_payload = {
            "status": "COMPLETED",
            "pairs": {pair_id: pair_payload},
        }
        # Old completed runs retain historical REUSE eligibility.
        (run_dir / "strategy_comparison.json").write_text(
            json.dumps(run_payload), encoding="utf-8"
        )
        legacy_hit = reuse._find_reusable_baseline_source(
            root=root, settings=settings, status=status, off_arm=off_arm
        ) == pair_dir

        # A newer interrupted run may have complete-looking pair-local files,
        # but no valid completed run-level JSON. It must never crash planning,
        # nor shadow an older eligible C96 baseline.
        interrupted_run = runs_root / "20990102_000000_interrupted_synthetic"
        interrupted_pair = interrupted_run / "pairs" / pair_id
        interrupted_pair.mkdir(parents=True)
        for name in required_names:
            (interrupted_pair / name).write_text(
                json.dumps(pair_payload) if name.endswith(".json") else "baseline\n",
                encoding="utf-8",
            )
        interrupted_report = interrupted_run / "strategy_comparison.json"
        interrupted_candidates_skipped = True
        for invalid_publication in (None, "broken json", "null", "[]", '{"status": "RUNNING"}'):
            if invalid_publication is None:
                interrupted_report.unlink(missing_ok=True)
            else:
                interrupted_report.write_text(invalid_publication, encoding="utf-8")
            interrupted_candidates_skipped &= (
                reuse._find_reusable_baseline_source(
                    root=root, settings=settings, status=status, off_arm=off_arm
                ) == pair_dir
            )

        integrity = build_completed_pair_file_integrity({pair_id: pair_dir})
        run_payload[PAIR_FILE_INTEGRITY_FIELD] = integrity
        run_report = run_dir / "strategy_comparison.json"
        run_report.write_text(json.dumps(run_payload), encoding="utf-8")
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({PAIR_FILE_INTEGRITY_FIELD: integrity}), encoding="utf-8"
        )
        pinned_hit = reuse._find_reusable_baseline_source(
            root=root, settings=settings, status=status, off_arm=off_arm
        ) == pair_dir
        trades = pair_dir / "no_filter_trades.csv"
        trades.write_text("corrupted\n", encoding="utf-8")
        corrupted_miss = reuse._find_reusable_baseline_source(
            root=root, settings=settings, status=status, off_arm=off_arm
        ) is None
        from services.research.strategy_compare_replay import _load_reusable_no_filter_baseline
        try:
            _load_reusable_no_filter_baseline(
                pair_dir, comparison_mode="score-ranking",
                expected_dataset=metadata["dataset"],
                expected_params_sha256=metadata["params_file_sha256"],
                expected_param_policy=metadata["requested_param_policy"],
                expected_param_evaluation_mode=metadata["param_evaluation_mode"],
                expected_optional_entry_filter_policy=metadata["optional_entry_filter_policy"],
                expected_shared_param_overrides=metadata["shared_param_overrides"],
                expected_max_positions=metadata["max_positions"],
                expected_enable_rotation=metadata["enable_rotation"],
                expected_start_date=period["start"],
                expected_end_date=period["end"],
            )
        except ValueError as exc:
            read_boundary_rejected_corrupt_csv = "完整性" in str(exc)
        else:
            read_boundary_rejected_corrupt_csv = False
        trades.write_text("baseline\n", encoding="utf-8")
        manifest_path.unlink()
        missing_manifest_miss = reuse._find_reusable_baseline_source(
            root=root, settings=settings, status=status, off_arm=off_arm
        ) is None

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "shared_baseline_cache_pins_csv_content_but_preserves_legacy_compatibility",
        True, legacy_hit and pinned_hit and corrupted_miss
        and read_boundary_rejected_corrupt_csv and missing_manifest_miss,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "shared_baseline_skips_missing_malformed_or_incomplete_run_publications",
        True, interrupted_candidates_skipped,
    )


def append_model_robustness_pit_strategy_reuse_contract_checks(*, results, case_id: str) -> None:
    """A model-robustness score may be reused, but strategy results remain separate."""

    from types import SimpleNamespace
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    from core.strategy_compare_policy import (
        get_strategy_comparison_settings,
        get_strategy_multi_seed_robustness_settings,
    )
    from filters.breakout_quality.paths import resolve_model_research_robustness_seed_dirs
    from filters.breakout_quality import strategy_compare_pit_contract as pit_contract
    from filters.breakout_quality.point_in_time_schedule import FOLD_CALENDAR_ANCHOR
    from services.research import breakout_quality_application as model_app
    from services.research import strategy_compare_training as training
    from services.research import strategy_multi_seed_robustness as robust

    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
    # AI: Exercise REUSE with a currently authorized model source, not an
    # archived identity that is still present in the compatibility catalog.
    active_source_arms = tuple(
        arm for arm in settings.enabled_arms
        if arm.dl_enabled and arm.dl_id in settings.dl_sources
    )
    # This case tests the *implicit* canonical anchor; choose by capability,
    # not by the order in which enabled strategy arms happen to be registered.
    source = next(
        settings.dl_sources[arm.dl_id]
        for arm in active_source_arms
        if settings.dl_sources[arm.dl_id].point_in_time_fold_anchor_date in (None, "")
        and not settings.dl_sources[arm.dl_id].point_in_time_single_score_block
    )
    same_source_arms = tuple(
        arm for arm in active_source_arms if arm.dl_id == source.dl_id
    )
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(source.experiment_profile)
    )
    seed = 693545351
    # Current producer persists the effective default calendar anchor, NOT
    # null, even when rolling mode omitted --fold-anchor-date. This is the
    # historical 44/44 false-TRAIN regression.
    effective_anchor = FOLD_CALENDAR_ANCHOR.strftime("%Y-%m-%d")
    natural_manifest = {
        "fold_anchor_date": effective_anchor,
        "fold_identity": {
            "calendar_anchor": effective_anchor,
            "stable_when_earlier_folds_are_added": True,
        },
        "fold_months": int(source.point_in_time_fold_months or workflow.point_in_time_fold_months),
        "single_score_block": False,
        "score_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "folds": [{"selected_epoch": 1}],
    }
    # AI: PIT READY validates both the score schedule and the published FIT
    # version. Supply a real, isolated rolling fold publication so this test
    # cannot silently skip the version owner or depend on the user's artifacts.
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    fold_id = "fold_20210101_20211231"
    persisted_manifest = dict(
        natural_manifest,
        experiment_profile=str(source.experiment_profile),
        folds=[{"fold_id": fold_id, "selected_epoch": 1}],
    )
    with tempfile.TemporaryDirectory(prefix="strategy_pit_anchor_") as temp_dir:
        root = Path(temp_dir)
        publication = pit_contract.resolve_strategy_compare_selection_pit_bundle_dir(
            root=root, source=source
        )
        fold_dir = publication / "folds" / fold_id
        fold_dir.mkdir(parents=True)
        (publication / "selection_point_in_time_manifest.json").write_text(
            json.dumps(persisted_manifest), encoding="utf-8"
        )
        (fold_dir / "manifest.json").write_text(
            json.dumps({
                "experiment_profile": str(source.experiment_profile),
                "experiment_settings": get_breakout_quality_experiment_profile(
                    str(source.experiment_profile)
                ).as_manifest_payload(),
            }),
            encoding="utf-8",
        )
        with patch.object(
            pit_contract, "load_selection_point_in_time_ranking_contract",
            return_value=SimpleNamespace(seed=seed, manifest=dict(persisted_manifest)),
        ):
            try:
                loaded = pit_contract.load_validated_selection_pit_strategy_compare_contract(
                    root=root, source=source, workflow=workflow, seed=seed,
                    comparison_start="2021-01-01", comparison_end="2026-03-02",
                )
            except ValueError:
                loaded = None
    changed_anchor = dict(natural_manifest, fold_anchor_date="2020-12-01")
    changed_mode = dict(natural_manifest, fold_identity=dict(
        natural_manifest["fold_identity"], stable_when_earlier_folds_are_added=False,
    ))
    try:
        pit_contract.validate_selection_pit_fold_anchor_identity(
            changed_anchor, requested_fold_anchor_date=source.point_in_time_fold_anchor_date,
        )
    except ValueError as exc:
        changed_anchor_rejected = "fold_anchor_date mismatch" in str(exc)
    else:
        changed_anchor_rejected = False
    try:
        pit_contract.validate_selection_pit_fold_anchor_identity(
            changed_mode, requested_fold_anchor_date=source.point_in_time_fold_anchor_date,
        )
    except ValueError as exc:
        changed_mode_rejected = "fold_identity schedule mismatch" in str(exc)
    else:
        changed_mode_rejected = False
    explicit_manifest = dict(natural_manifest, fold_identity=dict(
        natural_manifest["fold_identity"], stable_when_earlier_folds_are_added=False,
    ))
    try:
        pit_contract.validate_selection_pit_fold_anchor_identity(
            explicit_manifest, requested_fold_anchor_date=effective_anchor,
        )
        explicit_accepted = True
    except ValueError:
        explicit_accepted = False
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "canonical_default_fold_anchor_matches_persisted_producer_and_rejects_wrong_schedule",
        True, loaded is not None and changed_anchor_rejected
        and changed_mode_rejected and explicit_accepted,
    )
    with tempfile.TemporaryDirectory(prefix="model_robustness_reuse_") as temporary:
        root = Path(temporary)
        expected_dirs = resolve_model_research_robustness_seed_dirs(
            root, source.filter_id, source.model_architecture,
            source.experiment_profile, seed=seed, rolling=True,
        )
        with patch.object(model_app, "PROJECT_ROOT", root):
            producer_dirs = model_app._model_robustness_dirs(
                SimpleNamespace(
                    filter_id=source.filter_id,
                    model_architecture=source.model_architecture,
                    experiment_profile=source.experiment_profile,
                ), seed=seed, rolling=True,
            )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "model_and_strategy_resolve_one_canonical_robustness_seed_namespace",
            True, producer_dirs == expected_dirs,
        )
        model_dir, research_dir = expected_dirs
        model_dir.mkdir(parents=True)
        for filename, content in (
            (training.SELECTION_POINT_IN_TIME_SCORE_FILENAME, "synthetic-score"),
            (training.SELECTION_POINT_IN_TIME_MANIFEST_FILENAME, json.dumps(natural_manifest)),
            (training.SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME, "synthetic-audit"),
        ):
            (model_dir / filename).write_text(content, encoding="utf-8")
        saved_bytes = {
            path.name: path.read_bytes() for path in model_dir.iterdir()
            if path.is_file()
        }
        identity = {
            "score": str(model_dir / training.SELECTION_POINT_IN_TIME_SCORE_FILENAME),
            "score_sha256": "synthetic-validated-score-hash",
            "selected_epoch": 7, "fold_count": 6,
            "score_execution_start": "2021-01-01",
            "score_available_from": "2021-01-01",
            "score_available_through": "2026-03-02",
        }
        def validation_fixture(**_kwargs):
            persisted = json.loads(
                (model_dir / training.SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            pit_contract.validate_selection_pit_fold_anchor_identity(
                persisted, requested_fold_anchor_date=source.point_in_time_fold_anchor_date,
            )
            return dict(identity)

        with patch.object(training, "validate_strategy_compare_training_artifacts", side_effect=validation_fixture) as validation, patch.object(
            training, "pit_fold_fitting_settings_issues", return_value=()
        ) as fitting:
            evidence, reason = training.inspect_reusable_model_research_rolling_score(
                project_root=root, source=source, workflow=workflow, seed=seed,
                comparison_start="2021-01-01", comparison_end="2026-03-02",
            )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "strategy_reuses_readonly_research_robustness_pit_score_after_strict_validation",
            True, evidence is not None and reason == "READY"
            and evidence.get("model_artifact_origin") == "model_research_rolling_robustness"
            and validation.call_args.kwargs["seed"] == seed
            and validation.call_args.kwargs["point_in_time_dir_override"] == model_dir
            and validation.call_args.kwargs["comparison_end"] == "2026-03-02"
            and fitting.called
            and all((model_dir / name).read_bytes() == payload for name, payload in saved_bytes.items()),
        )
        with patch.object(
            training, "validate_strategy_compare_training_artifacts",
            side_effect=ValueError("PIT score SHA256 mismatch")
        ):
            invalid, invalid_reason = training.inspect_reusable_model_research_rolling_score(
                project_root=root, source=source, workflow=workflow, seed=seed,
                comparison_start="2021-01-01", comparison_end="2026-03-02",
            )
        with patch.object(training, "validate_strategy_compare_training_artifacts", return_value=dict(identity)), patch.object(
            training, "pit_fold_fitting_settings_issues", return_value=("epochs stale",)
        ):
            stale, stale_reason = training.inspect_reusable_model_research_rolling_score(
                project_root=root, source=source, workflow=workflow, seed=seed,
                comparison_start="2021-01-01", comparison_end="2026-03-02",
            )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "model_robustness_sha_or_fitting_contract_mismatch_is_never_reused",
            True, invalid is None and "SHA256" in invalid_reason
            and stale is None and "epochs stale" in stale_reason,
        )
        # An anchored Research score is not the same score-universe identity
        # as a non-anchored Strategy Compare source, even if its date coverage fits.
        manifest_path = model_dir / training.SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
        manifest_path.write_text('{"folds": [], "fold_anchor_date": "2020-12-01"}', encoding="utf-8")
        with patch.object(
            training, "validate_strategy_compare_training_artifacts", side_effect=validation_fixture
        ), patch.object(training, "pit_fold_fitting_settings_issues", return_value=()):
            anchored, anchored_reason = training.inspect_reusable_model_research_rolling_score(
                project_root=root, source=source, workflow=workflow, seed=seed,
                comparison_start="2021-01-01", comparison_end="2026-03-02",
            )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "research_rolling_score_reuse_requires_exact_fold_anchor_identity",
            True, anchored is None and "fold_anchor_date mismatch" in anchored_reason,
        )
        manifest_path.write_text('{"folds": []}', encoding="utf-8")
        (model_dir / training.SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME).unlink()
        missing, missing_reason = training.inspect_reusable_model_research_rolling_score(
            project_root=root, source=source, workflow=workflow, seed=seed,
            comparison_start="2021-01-01", comparison_end="2026-03-02",
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "model_robustness_missing_audit_cannot_claim_reuse",
            True, missing is None and "missing" in missing_reason,
        )

        # A preexisting Research score must bypass subprocess fitting, without
        # changing any Research [1] file or falsely completing strategy replay.
        job = {
            "settings": settings, "dl_id": str(source.dl_id), "seed": seed,
            "model_dir": str(root / "strategy_local_model_absent"),
            "research_dir": str(root / "strategy_local_research_absent"),
            "comparison_start": "2021-01-01",
            "comparison_end": "2026-03-02", "reuse_completed": True,
        }
        with patch.object(
            robust, "inspect_reusable_model_research_rolling_score",
            return_value=(dict(identity, model_artifact_origin="model_research_rolling_robustness"), "READY"),
        ), patch.object(
            robust, "run_strategy_compare_training_unit",
            side_effect=AssertionError("must not refit completed research model"),
        ) as trainer, patch.object(
            robust, "_prepare_trade_path_teacher_for_model",
            side_effect=AssertionError("must not prepare teachers for completed Research score"),
        ) as teacher_preparer:
            reused = robust._train_one_unit(job)
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_training_unit_short_circuits_to_existing_research_model_score",
            True, reused["reused"] is True and not trainer.called
            and not teacher_preparer.called
            and reused["artifacts"].get("model_artifact_origin") == "model_research_rolling_robustness"
            and not Path(job["model_dir"]).exists(),
        )

        # AI: A historical MinTrade teacher is a *seed-scoped* Optimizer input.
        # CN/CO consume the same teacher within each seed; completed scientific
        # results do not trigger a new teacher. A missing source is BUILD only
        # under the canonical Producer, never silently substituted from seed 1.
        from services.breakout_quality.trade_path_target_source import (
            registered_trade_path_source,
        )
        benchmark_seeds = get_strategy_multi_seed_robustness_settings(
            settings.profile_id
        ).resolved_seeds
        missing_seed = int(benchmark_seeds[1])
        produced = set()
        executed = []

        def inspect_p16(*, project_root, seed):
            return {"action": "REUSE", "reason": "same seed P16 verified",
                    "path": Path(project_root) / f"p16_{seed}.json"}

        def inspect_history(*, project_root, seed):
            return {"action": "REUSE" if seed in produced or seed == 693545351 else "BUILD",
                    "reason": "same seed historical evidence",
                    "path": Path(project_root) / f"history_{seed}.json"}

        def execute_history(*, project_root, seed):
            executed.append(seed)
            produced.add(seed)

        p16 = SimpleNamespace(name="formal_mintrade40_p16", priority=30,
                              builder_type="optimizer_formal_mintrade40_p16",
                              inspect=inspect_p16, execute=lambda **_kwargs: None)
        history = SimpleNamespace(name="formal_mintrade40_history", priority=40,
                                  builder_type="optimizer_formal_mintrade40_history",
                                  inspect=inspect_history, execute=execute_history)
        real_path_arms = tuple(
            arm for arm in settings.enabled_arms
            if arm.dl_id in settings.dl_sources
            and registered_trade_path_source(
                settings.dl_sources[arm.dl_id].experiment_profile
            ) is not None
        )
        teacher_model_profile = str(
            settings.dl_sources[real_path_arms[0].dl_id].experiment_profile
        )
        completed_first_seed = {
            (arm.arm_id, seed): {"action": robust.RESULT_ACTION_REUSE}
            for arm in real_path_arms
        }
        with patch.object(robust, "PROJECT_ROOT", root), patch.object(
            robust, "_trade_path_teacher_requirements", return_value=(p16, history)
        ):
            teacher_rows, teacher_blockers = robust._trade_path_seed_plan(
                cfg=SimpleNamespace(resolved_seeds=(seed, missing_seed)),
                settings=settings, model_arms=real_path_arms,
                robustness_unit_states=completed_first_seed,
                model_source_reuse={},
            )
            robust._prepare_trade_path_teacher_for_model(
                experiment_profile=teacher_model_profile,
                seed=missing_seed, authorized=True,
            )
            robust._prepare_trade_path_teacher_for_model(
                experiment_profile=teacher_model_profile,
                seed=missing_seed, authorized=True,
            )
            refreshed_rows, refreshed_blockers = robust._trade_path_seed_plan(
                cfg=SimpleNamespace(resolved_seeds=(seed, missing_seed)),
                settings=settings, model_arms=real_path_arms,
                robustness_unit_states=completed_first_seed,
                model_source_reuse={},
            )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_missing_historical_min_seed_planned_once_for_cm_cn_co",
            True, len(real_path_arms) >= 3 and not teacher_blockers
            and len(teacher_rows) == 2
            and {item[0] for item in teacher_rows} == {"REUSE", "BUILD"}
            and all(f"seed={missing_seed}" in item[1] for item in teacher_rows)
            and not refreshed_blockers and all(item[0] == "REUSE" for item in refreshed_rows),
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_teacher_build_is_idempotent_and_seed_isolated",
            True, executed == [missing_seed] and seed not in executed,
        )
        with patch.object(robust, "_trade_path_teacher_requirements", return_value=(
            SimpleNamespace(name="formal_mintrade40_history", priority=40,
                            builder_type="optimizer_formal_mintrade40_history",
                            inspect=lambda **_kwargs: {"action": "BLOCKED", "reason": "invalid original preflight"},
                            execute=execute_history),
        )):
            try:
                robust._prepare_trade_path_teacher_for_model(
                    experiment_profile=teacher_model_profile, seed=missing_seed,
                    authorized=True,
                )
            except RuntimeError as exc:
                blocked_preserved = "BLOCKED" in str(exc) and "invalid original preflight" in str(exc)
            else:
                blocked_preserved = False
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_invalid_historical_teacher_preserves_bytes_without_fallback",
            True, blocked_preserved and executed == [missing_seed],
        )
        resumable_model = root / "incomplete_isolated_score"
        resumable_model.mkdir()
        existing_fold = resumable_model / "fold_checkpoint.bin"
        existing_fold.write_bytes(b"preserve-validated-fitted-evidence")
        cold_job = {
            **job, "reuse_completed": False, "model_dir": str(resumable_model),
            "research_dir": str(root / "not_started_training"),
        }
        with patch.object(
            robust, "_prepare_trade_path_teacher_for_model",
            side_effect=RuntimeError("seed historical source BLOCKED"),
        ) as teacher_guard, patch.object(
            robust, "run_strategy_compare_training_unit",
            side_effect=AssertionError("must not run PIT trainer before teacher verification"),
        ) as cold_trainer:
            try:
                robust._train_one_unit(cold_job)
            except RuntimeError as exc:
                early_blocked = "historical source BLOCKED" in str(exc)
            else:
                early_blocked = False
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_teacher_preflight_blocks_before_deleting_resumable_folds",
            True, early_blocked and teacher_guard.call_count == 1
            and not cold_trainer.called
            and existing_fold.read_bytes() == b"preserve-validated-fitted-evidence",
        )

        cfg = get_strategy_multi_seed_robustness_settings("broker_cash_dl_conversion_rolling")
        shared_root = Path(robust.PROJECT_ROOT / str(cfg.checkpoint_cache_root)).resolve()
        units = robust._training_units(
            seeds=(seed,), stochastic_arms=same_source_arms,
            settings=settings, completed=set(), model_root=root / "strategy_model",
            run_root=root / "strategy_run", comparison_start="2021-01-01",
            comparison_end="2026-03-02", reuse_completed=True,
            benchmark_id=cfg.benchmark_id, checkpoint_cache_root=cfg.checkpoint_cache_root,
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "strategy_checkpoint_cache_uses_research_shared_fitting_identity_root_not_seed_silo",
            True, len(units) == 1
            and len(units[0]["replay_arms"]) == len(same_source_arms)
            and Path(units[0]["checkpoint_cache_root"]) == shared_root,
        )

        # AI: Verify execution-plan REUSE against the current dynamic arm set,
        # rather than a frozen historic AO/CG/CH arm count.
        status = {
            "preparation_plan": SimpleNamespace(actions=()),
            "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
            "artifact_identities": {},
        }
        fixed_arms, strategy_arms = robust._robustness_arms(settings, cfg)
        ready_contract = {"fingerprint": "synthetic", "resolved_seeds": cfg.resolved_seeds}
        # The READY contract must have one verified benchmark parameter per
        # declared arm/seed, even when the test is focused only on model scores.
        # Do not bypass the production inventory guard with an empty fixture.
        signed_params = root / "signed_params.json"
        signed_manifest = root / "signed_manifest.json"
        signed_params.write_text("{}", encoding="utf-8")
        signed_manifest.write_text("{}", encoding="utf-8")
        verified_params = {
            (arm.arm_id, int(seed)): {
                "path": str(signed_params), "manifest_path": str(signed_manifest),
                "family": "min", "evaluation_mode": "rolling",
                "param_policy": "base-finalist-best", "plan_action": "REUSE",
            }
            for arm in strategy_arms for seed in cfg.resolved_seeds
        }
        with patch.object(
            robust, "_discover_registered_strategy_only_scientific_reuse", return_value={}
        ), patch(
            "services.research.position_model_dependency.inspect_holding_model",
            return_value={"ready": True, "path": str(root / "holding_manifest.json"),
                          "reason": "isolated verified position dependency", "sha256": "c" * 64},
        ), missing_teacher_artifact_fixture(robust), patch.object(
            robust, "_benchmark_parameter_plan_rows", return_value=([], [], verified_params)
        ), patch.object(
            robust, "compute_strategy_param_scientific_sha256", return_value="a" * 64
        ), patch.object(
            robust, "resolve_strategy_param_evaluation_identity_sha256", return_value="b" * 64
        ), patch.object(
            robust, "build_multi_seed_robustness_contract", return_value=ready_contract
        ), patch.object(
            robust, "_resolve_existing_scientific_unit_states", return_value=(None, None, {})
        ), patch.object(robust, "_run_root", return_value=root / "empty_results"):
            with patch.object(
                robust, "inspect_reusable_model_research_rolling_score",
                return_value=({"score": "valid"}, "READY"),
            ):
                reused_plan, reused_state = robust._render_robustness_execution_plan(
                    cfg=cfg, settings=settings, status=status,
                    fixed_arms=fixed_arms, stochastic_arms=strategy_arms,
                )
            with patch.object(
                robust, "inspect_reusable_model_research_rolling_score",
                return_value=(None, "score/manifest missing"),
            ):
                missing_plan, missing_state = robust._render_robustness_execution_plan(
                    cfg=cfg, settings=settings, status=status,
                    fixed_arms=fixed_arms, stochastic_arms=strategy_arms,
                )
        model_replay_arms = tuple(
            arm for arm in strategy_arms if arm.dl_enabled
        )
        # A replay arm may consume multiple PIT sources (for example, rank
        # plus an auxiliary eligibility head). Count distinct physical model
        # source/seed units, not only the primary arm.dl_id. Sources shared by
        # several replay arms must still be counted just once per seed.
        from filters.breakout_quality.strategy_compare_dl_artifacts import (
            resolve_arm_runtime_dl_source_ids,
        )
        required_source_ids = {
            dl_id
            for arm in model_replay_arms
            for dl_id in resolve_arm_runtime_dl_source_ids(settings, arm)
        }
        grouped_source_ids = {
            dl_id
            for dl_id, _entries in robust._training_source_groups(
                model_replay_arms, settings=settings
            )
        }
        expected_reused_scores = len(required_source_ids) * len(cfg.resolved_seeds)
        reused_plan = "".join(reused_plan.splitlines())
        missing_plan = "".join(missing_plan.splitlines())
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "active_arm_plan_reports_model_reuse_and_only_replays_new_strategy_scientific_units",
            True, reused_state["overall_status"] == "READY"
            and required_source_ids == grouped_source_ids
            and len(reused_state["model_source_artifacts"]) == expected_reused_scores
            and all(
                state["action"] != "REUSE"
                for state in reused_state["robustness_unit_states"].values()
            )
            and "0 待PIT／checkpoint確認" in reused_plan
            and f"Research模型score REUSE={expected_reused_scores}" in reused_plan
            and not missing_state["model_source_artifacts"]
            and f"{expected_reused_scores} 待PIT／checkpoint確認" in missing_plan
            and "Research模型score REUSE=0" in missing_plan,
        )

    append_bidirectional_model_robustness_score_reuse_checks(results=results, case_id=case_id)
    append_model_unit_cross_run_scientific_reuse_checks(results=results, case_id=case_id)
    from tools.validate.synthetic_strategy_robustness_unit_cases import append_strategy_robustness_unit_checks
    append_strategy_robustness_unit_checks(results=results, case_id=case_id)



def append_bidirectional_model_robustness_score_reuse_checks(*, results, case_id: str) -> None:
    """[5] Rolling Robustness publishes the exact [3] model evidence namespace.

    A different period, source or benchmark cannot share scores, although the
    FIT owner may still import an identical fold checkpoint. No real trainer,
    optimizer or score calculation is invoked by these isolated cases.
    """
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    from core.strategy_compare_policy import (
        get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
    )
    from filters.breakout_quality.paths import resolve_model_research_robustness_seed_dirs
    from filters.breakout_quality.strategy_compare_pit_contract import (
        StrategyComparePITPeriodMismatch,
        validate_selection_pit_score_period,
    )
    from services.breakout_quality import fitted_model_artifacts
    from services.research import strategy_compare_training as trainer
    from services.research import strategy_multi_seed_robustness as robust

    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
    cfg = get_strategy_multi_seed_robustness_settings("broker_cash_dl_conversion_rolling")
    arms = tuple(arm for arm in settings.enabled_arms if arm.dl_enabled)
    source = next(
        settings.dl_sources[arm.dl_id] for arm in arms
        if trainer.resolve_shared_model_rolling_robustness_dirs(
            project_root=Path.cwd(), source=settings.dl_sources[arm.dl_id],
            workflow=get_breakout_quality_workflow_settings(
                experiment_profile=settings.dl_sources[arm.dl_id].experiment_profile,
            ), seed=int(cfg.resolved_seeds[0]), benchmark_id=cfg.benchmark_id,
            comparison_start="2021-01-01",
        ) is not None
    )
    source_arms = tuple(arm for arm in arms if arm.dl_id == source.dl_id)
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=source.experiment_profile,
    )
    seed = int(cfg.resolved_seeds[0])
    with tempfile.TemporaryDirectory(prefix="bidirectional_robustness_reuse_") as td:
        root = Path(td)
        canonical = resolve_model_research_robustness_seed_dirs(
            root, source.filter_id, source.model_architecture,
            source.experiment_profile, seed=seed, rolling=True,
        )
        with patch.object(robust, "PROJECT_ROOT", root):
            units = robust._training_units(
                seeds=(seed,), stochastic_arms=source_arms,
                settings=settings, completed=set(),
                model_root=root / "strategy_models", run_root=root / "strategy_outputs",
                comparison_start="2021-01-01", comparison_end="2026-03-02",
                reuse_completed=True, benchmark_id=cfg.benchmark_id,
                checkpoint_cache_root=None,
            )
        unit = units[0]
        add_check(results, "synthetic_breakout_quality", case_id,
                  "strategy_robustness_publishes_into_model_robustness_canonical_seed_root",
                  True, len(units) == 1
                  and unit["canonical_model_rolling_robustness"] is True
                  and Path(unit["model_dir"]) == canonical[0]
                  and Path(unit["research_dir"]) == canonical[1]
                  and Path(unit["transient_model_dir"]) != canonical[0],
        )
        rejected = trainer.resolve_shared_model_rolling_robustness_dirs(
            project_root=root, source=source, workflow=workflow, seed=seed,
            benchmark_id="another_benchmark", comparison_start="2021-01-01",
        )
        unlisted = trainer.resolve_shared_model_rolling_robustness_dirs(
            project_root=root, source=source, workflow=workflow,
            seed=1234567, benchmark_id=cfg.benchmark_id,
            comparison_start="2021-01-01",
        )
        from dataclasses import replace as dc_replace
        forward = dc_replace(source, point_in_time_single_score_block=True)
        unshared_forward = trainer.resolve_shared_model_rolling_robustness_dirs(
            project_root=root, source=forward, workflow=workflow, seed=seed,
            benchmark_id=cfg.benchmark_id, comparison_start="2021-01-01",
        )
        add_check(results, "synthetic_breakout_quality", case_id,
                  "canonical_model_score_sharing_rejects_wrong_benchmark_seed_and_forward",
                  True, rejected is None and unlisted is None and unshared_forward is None)

        canonical[0].mkdir(parents=True)
        canonical[1].mkdir(parents=True)
        local_dir = Path(unit["transient_model_dir"])
        local_research = Path(unit["transient_research_dir"])
        local_dir.mkdir(parents=True)
        local_research.mkdir(parents=True)
        (canonical[0] / "shared_evidence.bin").write_bytes(b"CANONICAL")
        (local_dir / "transient_score.bin").write_bytes(b"ISOLATED")
        robust._cleanup_unit_artifacts(
            model_dir=local_dir, research_dir=local_research,
            keep_checkpoints=False, keep_scores=False,
            score_source=str(source.score_source),
        )
        add_check(results, "synthetic_breakout_quality", case_id,
                  "strategy_transient_cleanup_preserves_published_model_robustness",
                  True, (canonical[0] / "shared_evidence.bin").read_bytes() == b"CANONICAL"
                  and not local_dir.exists(),
        )

        mismatch_manifest = {"score_period": {"start": "2021-01-01", "end": "2026-04-01"}}
        try:
            validate_selection_pit_score_period(
                mismatch_manifest, comparison_start="2021-01-01",
                comparison_end="2026-03-02",
            )
        except StrategyComparePITPeriodMismatch:
            typed_mismatch = True
        else:
            typed_mismatch = False
        try:
            validate_selection_pit_score_period(
                {**mismatch_manifest,
                 "evaluation_policy": {"score_end_resolution": "manual"},
                 "available_history_period": {"end": "2026-04-01"}},
                comparison_start="2021-01-01", comparison_end="auto",
            )
        except StrategyComparePITPeriodMismatch:
            auto_resolution_isolated = True
        except ValueError:
            auto_resolution_isolated = False
        else:
            auto_resolution_isolated = True
        add_check(results, "synthetic_breakout_quality", case_id,
                  "explicit_period_mismatch_is_isolatable_but_corrupt_auto_resolution_is_not",
                  True, typed_mismatch and not auto_resolution_isolated)

        ready = {"score": "verified", "fold_count": 2, "score_sha256": "sha"}
        job = {
            **unit, "trainer_registry": {}, "trainer_registry_lock": None,
            "trainer_key": "synthetic", "seed_order": 1,
        }
        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("CURRENT", "OK")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", return_value=dict(ready)
        ) as validator, patch.object(
            robust, "run_strategy_compare_training_unit", side_effect=AssertionError("must reuse")
        ) as fit, patch.object(
            robust, "_prepare_trade_path_teacher_for_model", side_effect=AssertionError("must reuse")
        ) as upstream:
            reused = robust._train_one_unit(job)
        add_check(results, "synthetic_breakout_quality", case_id,
                  "strategy_robustness_reuses_completed_canonical_model_score_without_refit",
                  True, reused["reused"] is True
                  and validator.call_args.kwargs["model_dir"] == canonical[0]
                  and not fit.called and not upstream.called)

        # Missing canonical score is published by the same official PIT producer,
        # without modifying the comparison or DL scientific identity.
        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("MISSING", "absent")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", side_effect=FileNotFoundError("score not built")
        ), patch.object(
            robust, "inspect_reusable_model_research_rolling_score", return_value=(None, "not built")
        ), patch.object(
            robust, "_prepare_trade_path_teacher_for_model"
        ) as upstream, patch.object(
            robust, "run_strategy_compare_training_unit", return_value={"artifacts": dict(ready), "elapsed_sec": 0}
        ) as fit:
            published = robust._train_one_unit(job)
        add_check(results, "synthetic_breakout_quality", case_id,
                  "new_strategy_robustness_score_is_published_with_model_mode_and_same_seed",
                  True, published["reused"] is False
                  and fit.call_args.kwargs["canonical_model_rolling_robustness"] is True
                  and fit.call_args.kwargs["model_dir"] == canonical[0]
                  and fit.call_args.kwargs["seed"] == seed
                  and upstream.called)

        # An existing full Research score with different explicit end must
        # remain untouched; a separately published Strategy score is legal.
        local_dir.mkdir(parents=True, exist_ok=True)
        validation_calls = []
        def validate_mismatch_then_local(**kwargs):
            validation_calls.append(kwargs["model_dir"])
            if kwargs["model_dir"] == canonical[0]:
                raise StrategyComparePITPeriodMismatch("explicit end differs")
            return dict(ready)
        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("CURRENT", "OK")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", side_effect=validate_mismatch_then_local
        ), patch.object(
            robust, "run_strategy_compare_training_unit", side_effect=AssertionError("no new training")
        ) as fit:
            local_reuse = robust._train_one_unit(job)
        add_check(results, "synthetic_breakout_quality", case_id,
                  "different_score_period_reuses_strategy_local_score_not_research_score",
                  True, local_reuse["reused"] is True and not fit.called
                  and validation_calls == [canonical[0], local_dir],
        )

        for child in list(local_dir.iterdir()):
            if child.is_file():
                child.unlink()
        local_dir.rmdir()
        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("CURRENT", "OK")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", side_effect=validate_mismatch_then_local
        ), patch.object(
            robust, "_prepare_trade_path_teacher_for_model"
        ), patch.object(
            robust, "run_strategy_compare_training_unit", return_value={"artifacts": dict(ready), "elapsed_sec": 0}
        ) as fit:
            local_build = robust._train_one_unit(job)
        add_check(results, "synthetic_breakout_quality", case_id,
                  "different_score_period_builds_only_separate_score_publication",
                  True, local_build["reused"] is False
                  and fit.call_args.kwargs["model_dir"] == local_dir
                  and fit.call_args.kwargs["canonical_model_rolling_robustness"] is False,
        )

        # An auto-resolved model score may diverge from Strategy's explicit
        # period only after the Producer completes. Preserve Research source
        # and rerun the Strategy evaluation in its private score namespace.
        def fake_train(**kwargs):
            if kwargs["canonical_model_rolling_robustness"]:
                raise StrategyComparePITPeriodMismatch("actual end differs")
            return {"artifacts": dict(ready), "elapsed_sec": 0}
        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("MISSING", "new")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", side_effect=FileNotFoundError("not ready")
        ), patch.object(
            robust, "inspect_reusable_model_research_rolling_score", return_value=(None, "not ready")
        ), patch.object(
            robust, "_prepare_trade_path_teacher_for_model"
        ), patch.object(
            robust, "run_strategy_compare_training_unit", side_effect=fake_train
        ) as fit:
            late_fallback = robust._train_one_unit(job)
        add_check(results, "synthetic_breakout_quality", case_id,
                  "auto_model_score_period_difference_falls_back_without_overwriting_research",
                  True, late_fallback["reused"] is False
                  and fit.call_count == 2
                  and fit.call_args_list[0].kwargs["model_dir"] == canonical[0]
                  and fit.call_args_list[1].kwargs["model_dir"] == local_dir,
        )

        with patch.object(fitted_model_artifacts, "published_fit_contract_version_state", return_value=("CURRENT", "OK")), patch.object(
            robust, "validate_strategy_compare_training_artifacts", side_effect=ValueError("score SHA corrupted")
        ), patch.object(
            robust, "run_strategy_compare_training_unit", side_effect=AssertionError("must BLOCK")
        ) as fit:
            try:
                robust._train_one_unit(job)
            except RuntimeError as exc:
                blocked = "identity/integrity" in str(exc)
            else:
                blocked = False
        add_check(results, "synthetic_breakout_quality", case_id,
                  "corrupt_same_identity_canonical_robustness_score_blocks_and_preserves_source",
                  True, blocked and not fit.called
                  and (canonical[0] / "shared_evidence.bin").read_bytes() == b"CANONICAL",
        )


def append_model_unit_cross_run_scientific_reuse_checks(*, results, case_id: str) -> None:
    """Old validated DL-arm observations survive a new arm in the suite, not score drift."""
    import gzip
    from types import SimpleNamespace
    from core.strategy_compare_policy import (
        get_strategy_comparison_settings, get_strategy_multi_seed_robustness_settings,
    )
    from filters.breakout_quality.strategy_compare_dl_artifacts import (
        resolve_arm_runtime_dl_source_ids,
    )
    from services.research import strategy_multi_seed_robustness as robust

    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
    cfg = get_strategy_multi_seed_robustness_settings("broker_cash_dl_conversion_rolling")
    model_arms = tuple(
        arm for arm in settings.enabled_arms if arm.dl_enabled
    )
    arm = model_arms[0]
    seed = int(cfg.resolved_seeds[0])
    ids = resolve_arm_runtime_dl_source_ids(settings, arm)
    contract_arm = {
        "arm_id": arm.arm_id, "param_source": arm.param_source,
        "param_policy": "base-finalist-best", "rule_policy": arm.rule_policy,
        "strategy_seed_sensitive": True, "model_seed_sensitive": True,
        "dl_id": arm.dl_id, "dl_runtime_mode": arm.dl_runtime_mode,
        "dl_runtime_options": dict(arm.dl_runtime_options or {}),
        "runtime_dl_sources": [
            {"dl_id": dl_id, "score_source": "selection_point_in_time"}
            for dl_id in ids
        ],
        "seed_pairing": "same_seed",
    }
    parameter = {
        "identity_schema": robust.STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
        "family": "min", "evaluation_mode": "rolling",
        "param_policy": "base-finalist-best", "sha256": "param-good",
    }
    common = {
        "scientific_contract_version": robust.ROBUSTNESS_SCIENTIFIC_CONTRACT_VERSION,
        "dataset": "synthetic", "dataset_identity": {"stock_days": 630589},
        "max_positions": 10, "rotation": "off",
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "benchmark_id": "end_to_end_v1", "strategy_trials_per_fold": 300,
        "seed_pairing": "same_seed",
        "benchmark_parameter_artifact_identities": {
            f"{arm.arm_id}:seed={seed}": parameter
        },
    }
    old = {**common, "fingerprint": "old", "stochastic_arms": [contract_arm]}
    current = {
        **common, "fingerprint": "new",
        "stochastic_arms": [contract_arm, {"arm_id": "new_arm", "model_seed_sensitive": True}],
    }
    old_signature = robust._strategy_only_unit_scientific_signature_from_contract(
        contract=old, arm_id=arm.arm_id, seed=seed
    )
    current_signature = robust._strategy_only_unit_scientific_signature_from_contract(
        contract=current, arm_id=arm.arm_id, seed=seed
    )
    changed_arm = dict(contract_arm, dl_runtime_options={"eligibility_threshold": 0.9})
    changed_signature = robust._strategy_only_unit_scientific_signature_from_contract(
        contract={**old, "stochastic_arms": [changed_arm]},
        arm_id=arm.arm_id, seed=seed,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "model_arm_result_identity_survives_unrelated_new_arm_but_not_rule_change",
        True, old_signature == current_signature != changed_signature,
    )

    def fake_model_research(*, source, **_kwargs):
        return ({
            "selected_epoch": 3, "fold_count": 6, "model_sha256": None,
            "score_sha256": f"score-{source.dl_id}",
            "score_available_from": "2021-01-01",
            "score_available_through": "2026-03-02",
        }, "READY")

    artifacts = {dl: fake_model_research(source=settings.dl_sources[dl])[0] for dl in ids}
    source_identities = robust._runtime_replay_source_identities(artifacts)
    score_identity = {
        "primary_score_sha256": artifacts[ids[0]]["score_sha256"],
        "runtime_source_identity_sha256": robust.canonical_json_sha256(source_identities),
        "runtime_source_identities": source_identities,
    }
    source_row = {
        "arm_id": arm.arm_id, "seed": seed,
        "score_sha256": score_identity["primary_score_sha256"],
        "runtime_source_identity_sha256": score_identity["runtime_source_identity_sha256"],
        "strategy_param_sha256": "param-good",
        "strategy_param_manifest_sha256": "param-manifest",
        "selected_epoch": 3, "fold_count": 6,
        "return_over_max_drawdown": 10.0,
    }
    source_yearly = pd.DataFrame([
        {"arm_id": arm.arm_id, "seed": seed, "year": 2021,
         "return_pct": 4.0, "is_complete_year": True},
    ])
    source_attr = {
        "schema_version": robust.ATTRIBUTION_SOURCE_SCHEMA_VERSION,
        "scientific_fingerprint": "old", "arm_id": arm.arm_id, "seed": seed,
        "training_identity": {
            "score_sha256": source_row["score_sha256"],
            "runtime_source_identity_sha256": source_row["runtime_source_identity_sha256"],
        },
        "runtime_training_identities": score_identity["runtime_source_identities"],
    }
    mini_cfg = SimpleNamespace(
        reuse_completed=True, benchmark_id="end_to_end_v1",
        model_seed_sensitive_arm_ids=(arm.arm_id,), resolved_seeds=(seed,),
    )
    with tempfile.TemporaryDirectory(prefix="model_seed_scientific_reuse_") as temp_dir:
        root = Path(temp_dir)
        old_root = root / "old"
        new_root = root / "new"
        old_root.mkdir()
        new_root.mkdir()
        validated = {
            "contract": old,
            "seed_frame": pd.DataFrame([source_row]),
            "seed_yearly_frame": source_yearly,
        }
        def discover(row, attribution):
            data = dict(validated, seed_frame=pd.DataFrame([row]))
            with patch.object(
                robust, "inspect_reusable_model_research_rolling_score",
                side_effect=fake_model_research,
            ), patch.object(
                robust, "_iter_registered_robustness_scientific_runs",
                return_value=iter(((old_root, data),)),
            ), patch.object(
                robust, "_read_attribution_unit_manifest", return_value=attribution,
            ):
                return robust._discover_registered_model_scientific_reuse(
                    settings=settings, robustness=mini_cfg,
                    current_contract=current, stochastic_arms=(arm,),
                )
        found = discover(source_row, source_attr)
        wrong_score = discover({**source_row, "score_sha256": "changed"}, source_attr)
        missing_attr = discover(source_row, None)
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "old_dl_arm_reuse_requires_matching_pit_score_and_verified_attribution",
            True, len(found) == 1 and not wrong_score and not missing_attr,
        )
        alternate_source = dict(source_attr, runtime_training_identities={"forged": {}})
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "old_dl_arm_reuse_rejects_incompatible_multisource_head_identity",
            True, not discover(source_row, alternate_source),
        )
        roles = ("trades", "equity", "daily_capacity", "selected_buys", "execution")
        old_attribution_dir = robust._attribution_unit_dir(old_root, arm.arm_id, seed)
        old_attribution_dir.mkdir(parents=True)
        files = {}
        for role in roles:
            old_file = old_attribution_dir / f"{role}.csv.gz"
            with gzip.open(old_file, "wb") as dest:
                dest.write(f"sample {role}".encode("utf-8"))
            files[role] = {
                "path": str(old_file.relative_to(root)).replace("\\", "/"),
                "sha256": robust.compute_file_sha256(old_file), "row_count": 1,
            }
        copy_item = {
            **next(iter(found.values())),
            "source_run_root": old_root,
            "source_attribution": dict(source_attr, files=files),
        }
        full_new = dict(current, robustness_id=cfg.robustness_id,
                        profile_id=settings.profile_id)
        with patch.object(robust, "PROJECT_ROOT", root):
            robust._reference_registered_model_attribution(
                source_item=copy_item, destination_root=new_root,
                contract=full_new, arm_id=arm.arm_id, seed=seed,
                arm_order=1, seed_order=1, arm_name=arm.name,
            )
            new_attr = robust._read_attribution_unit_manifest(
                new_root, arm_id=arm.arm_id, seed=seed,
                expected_fingerprint="new",
            )
            old_file = old_attribution_dir / "trades.csv.gz"
            old_file.write_bytes(b"tampered")
            corrupted = False
            try:
                robust._reference_registered_model_attribution(
                    source_item=copy_item, destination_root=root / "corrupt_run",
                    contract=full_new, arm_id=arm.arm_id, seed=seed,
                    arm_order=1, seed_order=1, arm_name=arm.name,
                )
            except ValueError:
                corrupted = True
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "reused_dl_trade_attribution_is_referenced_and_signed_without_portfolio_replay",
            True, new_attr is not None and corrupted,
        )
