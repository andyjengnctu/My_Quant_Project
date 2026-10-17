from __future__ import annotations

from .synthetic_breakout_quality_support import (
    Path,
    add_check,
    io,
    json,
    patch,
    pd,
    redirect_stdout,
    replace,
    tempfile,
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
        "forward_oos"
    )
    c44_contract = forward_score_reuse_base.arms["C44"]

    # A completed result is only a reusable candidate while its scientific Params
    # producer is still BUILD/REBUILD/RESUME.  The final REUSE/RUN decision must be
    # made after canonical parameter production completes and the plan is re-resolved.
    parent_child_settings = strategy_config.get_strategy_comparison_settings(
        "non_dl_production_multiseed_oos"
    )
    parent_param_sources = tuple(
        dict.fromkeys(arm.param_source for arm in parent_child_settings.enabled_arms)
    )
    pending_plan = reuse_owner_module.StrategyPreparationPlan.from_actions(
        reuse_owner_module.StrategyPreparationAction(
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
    ready_plan = reuse_owner_module.StrategyPreparationPlan.from_actions(
        reuse_owner_module.StrategyPreparationAction(
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
    broker_plan = reuse_owner_module.StrategyPreparationPlan.from_actions(
        reuse_owner_module.StrategyPreparationAction(
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
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "new_forward_arm_can_reuse_byte_identical_pair_pinned_frozen_score_without_model_rebuild",
        True,
        all(
            callable(getattr(score_reuse_module, name, None))
            for name in (
                "_completed_pair_pinned_continuous_score",
                "_historical_continuous_score_provenance_entries",
                "_apply_completed_pair_frozen_score_reuse",
            )
        )
        and c44_contract.dl_runtime_mode
        == STRATEGY_DL_RUNTIME_MODE_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL
        and dict(c44_contract.dl_runtime_options).get("constrained_solver")
        == "exact_branch_and_bound_v1",
    )

    forward_score_reuse_arms = dict(forward_score_reuse_base.arms)
    forward_score_reuse_arms["C36"] = replace(
        forward_score_reuse_arms["C36"], enabled=True
    )
    forward_score_reuse_arms["C44"] = replace(
        forward_score_reuse_arms["C44"], enabled=True
    )
    forward_score_reuse_settings = replace(
        forward_score_reuse_base, arms=forward_score_reuse_arms
    )
    with tempfile.TemporaryDirectory() as archived_score_temp:
        archived_root = Path(archived_score_temp).resolve()
        archived_score_path = archived_root / "archive" / "cont13e_scores.csv"
        archived_score_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "ticker": "2330",
                    "date": "2021-01-04",
                    "group_index": 0,
                    "model_score": 0.80,
                },
                {
                    "ticker": "2317",
                    "date": "2021-01-05",
                    "group_index": 0,
                    "model_score": 0.70,
                },
            ]
        ).to_csv(archived_score_path, index=False)
        archived_score_sha = score_reuse_module._file_sha256(archived_score_path)
        archived_run_dir = (
            archived_root
            / "outputs"
            / "strategy_compare"
            / "forward_oos"
            / "runs"
            / "archived"
        )
        archived_run_dir.mkdir(parents=True, exist_ok=True)
        archived_c36 = forward_score_reuse_settings.arms["C36"]
        archived_group_id = score_reuse_module._pair_group_id(
            param_source=archived_c36.param_source,
            rule_policy=archived_c36.rule_policy,
            dl_id=str(archived_c36.dl_id or ""),
            dl_runtime_mode=str(archived_c36.dl_runtime_mode or ""),
        )
        archived_run_payload = {
            "status": "COMPLETED",
            "settings": forward_score_reuse_settings.as_dict(),
            "artifact_identities": {
                "dl:CONT13E:forward_scores": {
                    "path": "archive/cont13e_scores.csv",
                    "sha256": archived_score_sha,
                }
            },
            "comparison_period": {
                "start": "2021-01-01",
                "end": "2021-01-05",
            },
            "pairs": {
                archived_group_id: {
                    "metadata": {
                        "comparison_period": {
                            "start": "2021-01-01",
                            "end": "2021-01-05",
                        },
                        "score_signal_coverage": {
                            "required_start": "2021-01-01",
                            "first_scored_event": "2021-01-04",
                            "available_through": "2021-01-05",
                        },
                    }
                }
            },
        }
        (archived_run_dir / "strategy_comparison.json").write_text(
            json.dumps(archived_run_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        archived_score_status = {
            "comparison_period": {
                "start": "2021-01-01",
                "end": "2021-01-05",
            },
            "dl_sources": {
                "CONT13E": {
                    "files": {
                        "forward_scores": {
                            "path": "current/canonical/missing_scores.csv",
                        }
                    }
                }
            },
        }
        archived_score_replay_cache = {
            "pairs": {
                "C36": {
                    "source_run_dir": str(archived_run_dir),
                    "source_pair_dir": str(
                        archived_run_dir / "pairs" / archived_group_id
                    ),
                    "source_group_id": archived_group_id,
                },
                "C44": None,
            }
        }
        archived_score_diagnostics = []
        archived_score_recovered = (
            score_reuse_module._completed_pair_pinned_continuous_score(
                root=archived_root,
                settings=forward_score_reuse_settings,
                status=archived_score_status,
                replay_cache=archived_score_replay_cache,
                dl_id="CONT13E",
                diagnostics=archived_score_diagnostics,
            )
        )
        archived_plan_actions = [
            score_reuse_module.StrategyPreparationAction(
                action_id=f"dl:CONT13E:{artifact_name}",
                artifact_key=f"dl:CONT13E:{artifact_name}",
                action="BLOCKED",
                builder_type=None,
                description="synthetic blocked source",
                path=(
                    "current/canonical/missing_scores.csv"
                    if artifact_name == "forward_scores"
                    else f"current/canonical/{artifact_name}"
                ),
                dependencies=(),
                producer_work_type="model_training",
            )
            for artifact_name in ("model", "manifest", "report", "forward_scores")
        ]
        archived_score_status["preparation_plan"] = (
            score_reuse_module.StrategyPreparationPlan.from_actions(
                archived_plan_actions
            )
        )
        archived_score_status["overall_status"] = "BLOCKED"
        archived_score_status["comparison_ready"] = False
        archived_score_status["comparison_period"] = None
        archived_score_status["comparison_period_source"] = "runtime_pending"
        archived_score_applied = (
            score_reuse_module._apply_completed_pair_frozen_score_reuse(
                root=archived_root,
                settings=forward_score_reuse_settings,
                status=archived_score_status,
                replay_cache=archived_score_replay_cache,
            )
        )
        historical_only_diagnostics = []
        historical_only_recovered = (
            score_reuse_module._completed_pair_pinned_continuous_score(
                root=archived_root,
                settings=forward_score_reuse_base,
                status=archived_score_status,
                replay_cache={"pairs": {"C44": None}},
                dl_id="CONT13E",
                diagnostics=historical_only_diagnostics,
            )
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_frozen_score_reuse_prefers_archived_identity_path_when_current_canonical_path_moved",
        True,
        isinstance(archived_score_recovered, dict)
        and archived_score_recovered.get("path_source")
        == "archived_artifact_identity"
        and archived_score_recovered.get("sha256") == archived_score_sha
        and archived_score_recovered.get("execution_start") == "2021-01-01"
        and archived_score_recovered.get("available_from") == "2021-01-04"
        and archived_score_recovered.get("available_through") == "2021-01-05"
        and "REUSE_OK:archived_artifact_identity" in archived_score_diagnostics
        and archived_score_applied.get("overall_status") == "READY"
        and archived_score_applied.get("comparison_ready") is True
        and archived_score_applied.get("comparison_period")
        == {"start": "2021-01-01", "end": "2021-01-05"}
        and archived_score_applied.get("comparison_period_source")
        == "dl_runtime_common_overlap"
        and archived_score_applied.get("continuous_score_overrides", {})
        .get("CONT13E", {})
        .get("path_source")
        == "archived_artifact_identity"
        and {
            action.artifact_key: action.action
            for action in archived_score_applied["preparation_plan"].actions
        }
        == {
            "dl:CONT13E:model": "NOT_REQUIRED",
            "dl:CONT13E:manifest": "NOT_REQUIRED",
            "dl:CONT13E:report": "NOT_REQUIRED",
            "dl:CONT13E:forward_scores": "REUSE",
        }
        and isinstance(historical_only_recovered, dict)
        and historical_only_recovered.get("path_source")
        == "archived_artifact_identity"
        and historical_only_recovered.get("sha256") == archived_score_sha
        and "REUSE_OK:archived_artifact_identity" in historical_only_diagnostics,
    )


    # Selection-PIT cache identity must track only score columns consumed by each arm.
    selection_settings = strategy_config.get_strategy_comparison_settings(
        "extending_window_oos"
    )
    selection_period = {"start": "2021-01-01", "end": "2026-03-02"}
    projection_artifacts = {
        "dl:CONT13R_ROLL:manifest": {"sha256": "manifest-v1"},
        "dl:CONT13R_ROLL:audit": {"sha256": "audit-v1"},
        "dl:CONT13R_ROLL:forward_scores": {
            "sha256": "whole-file-v1",
            "score_projection_sha256": {
                "breakout_quality_score": "primary-v1",
                "raw_safety_score": "safety-v1",
            },
        },
    }
    auxiliary_changed = json.loads(json.dumps(projection_artifacts))
    auxiliary_changed["dl:CONT13R_ROLL:manifest"]["sha256"] = "manifest-v2"
    auxiliary_changed["dl:CONT13R_ROLL:audit"]["sha256"] = "audit-v2"
    auxiliary_changed["dl:CONT13R_ROLL:forward_scores"]["sha256"] = "whole-file-v2"
    auxiliary_changed["dl:CONT13R_ROLL:forward_scores"]["score_projection_sha256"][
        "raw_safety_score"
    ] = "safety-v2"
    primary_changed = json.loads(json.dumps(auxiliary_changed))
    primary_changed["dl:CONT13R_ROLL:forward_scores"]["score_projection_sha256"][
        "breakout_quality_score"
    ] = "primary-v2"

    def projected_fp(arm_id, artifacts):
        return score_reuse_module._pair_cache_fingerprint_from_payload(
            settings_payload=selection_settings.as_dict(),
            artifact_identities=artifacts,
            comparison_period=selection_period,
            off_arm_payload=selection_settings.arms["C58"].as_dict(),
            on_arm_payload=selection_settings.arms[arm_id].as_dict(),
            engine_schema_version=score_reuse_module.STRATEGY_COMPARE_ENGINE_SCHEMA_VERSION,
            parameter_evaluation_sha256="param-sha",
        )

    c66_base = projected_fp("C66", projection_artifacts)
    c66_aux = projected_fp("C66", auxiliary_changed)
    c66_primary = projected_fp("C66", primary_changed)
    c70_base = projected_fp("C70", projection_artifacts)
    c70_aux = projected_fp("C70", auxiliary_changed)
    c70_primary = projected_fp("C70", primary_changed)
    c71_base = projected_fp("C71", projection_artifacts)
    c71_aux = projected_fp("C71", auxiliary_changed)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_pit_pair_identity_ignores_unconsumed_auxiliary_score_but_tracks_consumed_columns",
        True,
        c66_base == c66_aux
        and c70_base == c70_aux
        and c66_base != c66_primary
        and c70_base != c70_primary
        and c71_base != c71_aux,
    )

    from filters.breakout_quality.strategy_score_projection import (
        compute_score_projection_sha256s,
    )
    with tempfile.TemporaryDirectory() as projection_tmp:
        projection_root = Path(projection_tmp).resolve()
        score_path = projection_root / "models" / "cont13r_scores.csv"
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
        on_arm = selection_settings.arms["C66"]
        off_arm = selection_settings.arms["C58"]
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
            / "20260826_000000_legacy_c66"
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
            "dl:CONT13R_ROLL:manifest": {"path": "models/old_manifest.json", "sha256": "old-manifest"},
            "dl:CONT13R_ROLL:audit": {"path": "models/old_audit.json", "sha256": "old-audit"},
            "dl:CONT13R_ROLL:forward_scores": {"path": "models/cont13r_scores.csv", "sha256": "old-whole-file"},
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
            "dl:CONT13R_ROLL:manifest": {"path": "models/new_manifest.json", "sha256": "new-manifest"},
            "dl:CONT13R_ROLL:audit": {"path": "models/new_audit.json", "sha256": "new-audit"},
            "dl:CONT13R_ROLL:forward_scores": {
                "path": "models/cont13r_scores.csv",
                "sha256": "new-whole-file",
                "score_projection_sha256": projection_map,
            },
        }
        status = {
            "artifact_identities": current_artifacts,
            "comparison_period": selection_period,
            "resolved_arm_parameter_identities": {"C66": {"sha256": "param-sha"}},
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
        score_frame.loc[0, "breakout_quality_score"] = 0.91
        score_frame.to_csv(score_path, index=False, encoding="utf-8-sig")
        changed_projection_map = compute_score_projection_sha256s(
            score_path=score_path,
            score_source="selection_point_in_time",
            columns=("breakout_quality_score", "raw_safety_score"),
        )
        changed_current_artifacts = json.loads(json.dumps(current_artifacts))
        changed_current_artifacts["dl:CONT13R_ROLL:forward_scores"][
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


def append_completed_pair_cache_contract_checks(
    *,
    results,
    case_id: str,
    settings,
    ready_status: dict,
    mocked_pair_payload: dict,
) -> None:
    """Validate completed-pair cache identity, output completeness, and matrix reuse."""

    from services.research import strategy_comparison as strategy_comparison_module
    from filters.breakout_quality.strategy_compare_sources import (
        resolve_strategy_param_evaluation_identity_sha256,
    )

    with tempfile.TemporaryDirectory() as identity_tmp:
        identity_root = Path(identity_tmp)
        base_payload = {
            "schema_type": "active_param_ensemble",
            "mode": "rolling",
            "selector": "base-finalist-best",
            "params_ensemble_by_effective_date": {
                "2021-01-01": [{"member_index": 1, "params": {"risk_per_trade": 0.01, "high_length": 100}}],
                "2022-01-01": [{"member_index": 1, "params": {"risk_per_trade": 0.02, "high_length": 120}}],
            },
            "summary": {"oos_end_date": "2022-12-31"},
        }
        changed_payload = json.loads(json.dumps(base_payload))
        changed_payload["params_ensemble_by_effective_date"]["2022-01-01"][0]["params"]["high_length"] = 220
        path_a = identity_root / "a.json"
        path_b = identity_root / "b.json"
        path_a.write_text(json.dumps(base_payload), encoding="utf-8")
        path_b.write_text(json.dumps(changed_payload), encoding="utf-8")
        oos_a = resolve_strategy_param_evaluation_identity_sha256(
            path_a, evaluation_mode="oos", start_date="2021-01-01", end_date="2022-12-31"
        )
        oos_b = resolve_strategy_param_evaluation_identity_sha256(
            path_b, evaluation_mode="oos", start_date="2021-01-01", end_date="2022-12-31"
        )
        rolling_a = resolve_strategy_param_evaluation_identity_sha256(
            path_a, evaluation_mode="rolling", start_date="2021-01-01", end_date="2022-12-31"
        )
        rolling_b = resolve_strategy_param_evaluation_identity_sha256(
            path_b, evaluation_mode="rolling", start_date="2021-01-01", end_date="2022-12-31"
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_param_result_identity_is_evaluation_bound_not_physical_schedule_bound",
        True,
        oos_a == oos_b and rolling_a != rolling_b and oos_a != rolling_a,
    )

    cache_off = settings.arms.get("C3")
    cache_on = settings.arms.get("C17")
    if cache_off is None or cache_on is None or not cache_on.dl_id:
        return

    with tempfile.TemporaryDirectory() as required_tmp:
        required_names = {
            path.name
            for path in strategy_comparison_module._pair_cache_required_files(
                Path(required_tmp),
                on_arm=cache_on,
            )
        }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_cache_requires_human_readable_strategy_report",
        True,
        "strategy_comparison.md" in required_names
        and "strategy_comparison.json" in required_names,
    )

    cache_period = {"start": "2021-01-01", "end": "2025-12-22"}
    cache_artifacts = {
        "param:min_roos": {"path": "models/min.json", "sha256": "param-sha"},
        "dl:CONT12A:model": {
            "path": "models/cont12a.pt",
            "sha256": "model-sha",
        },
        "dl:CONT12A:manifest": {
            "path": "models/cont12a.json",
            "sha256": "manifest-sha",
        },
        "dl:CONT12A:forward_scores": {
            "path": "models/cont12a.csv",
            "sha256": "scores-sha",
        },
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
            json.dumps(cached_pair_payload), encoding="utf-8"
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
            json.dumps(
                {
                    "status": "COMPLETED",
                    "settings": stored_settings,
                    "artifact_identities": cache_artifacts,
                    "comparison_period": cache_period,
                    "pairs": {pair_group: cached_pair_payload},
                }
            ),
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
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_pair_cache_uses_replay_identity_not_whole_config_and_invalidates_on_score_sha",
        True,
        cache_hit is not None
        and current_fp == report_changed_fp
        and cache_miss_after_score_change,
    )

    historical_ids = {"C3", "C17", "C18", "C19", "C20", "C21", "C22"}
    if not historical_ids.issubset(set(settings.arms)):
        return
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
            json.dumps(mocked_pair_payload), encoding="utf-8"
        )
        reuse_status = dict(ready_status)
        reuse_status["resolved_parameter_paths"] = {
            "min_roos": "models/min_roos.json"
        }
        reuse_status["resolved_arm_parameter_paths"] = {
            arm.arm_id: "models/min_roos.json"
            for arm in historical_settings.enabled_arms
        }
        reuse_status["dl_sources"] = {
            str(arm.dl_id): {"ready": True}
            for arm in historical_settings.enabled_arms
            if arm.dl_enabled and arm.dl_id
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
                "min_roos::base-finalist-best::all_off": {
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
            cached_execution_payload = strategy_comparison_module.run_strategy_comparison(
                project_root=reuse_root,
                quiet=False,
                status=reuse_status,
                auto_prepare=False,
            )
        cache_actions = {
            key: value["action"]
            for key, value in cached_execution_payload["pair_execution"].items()
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
            )
            >= 2
            and all(
                (
                    reuse_root
                    / str(
                        cached_execution_payload["pair_execution"][arm_id][
                            "current_pair_dir"
                        ]
                    )
                    / "strategy_comparison.md"
                ).is_file()
                for arm_id in ("C17", "C18", "C19", "C20")
            )
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multi_model_matrix_reuses_c17_c20_and_runs_only_c21_c22_with_shared_baseline",
        True,
        only_new_model_pairs_run,
    )


__all__ = [
    "append_completed_pair_cache_contract_checks",
    "append_completed_pair_score_reuse_contract_checks",
]


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

    settings = get_strategy_comparison_settings("forward_oos")
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
    from services.research import breakout_quality_application as model_app
    from services.research import strategy_compare_training as training
    from services.research import strategy_multi_seed_robustness as robust

    settings = get_strategy_comparison_settings("broker_cash_dl_conversion_rolling")
    source = settings.dl_sources["CONT13AO_ROLL"]
    workflow = get_breakout_quality_workflow_settings(
        experiment_profile=str(source.experiment_profile)
    )
    seed = 693545351
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
            (training.SELECTION_POINT_IN_TIME_MANIFEST_FILENAME, '{"folds": []}'),
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
        with patch.object(training, "validate_strategy_compare_training_artifacts", return_value=dict(identity)) as validation, patch.object(
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
            training, "validate_strategy_compare_training_artifacts", return_value=dict(identity)
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
            "settings": settings, "dl_id": "CONT13AO_ROLL", "seed": seed,
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
        ) as trainer:
            reused = robust._train_one_unit(job)
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "robustness_training_unit_short_circuits_to_existing_research_model_score",
            True, reused["reused"] is True and not trainer.called
            and reused["artifacts"].get("model_artifact_origin") == "model_research_rolling_robustness"
            and not Path(job["model_dir"]).exists(),
        )

        cfg = get_strategy_multi_seed_robustness_settings("broker_cash_dl_conversion_rolling")
        shared_root = Path(robust.PROJECT_ROOT / str(cfg.checkpoint_cache_root)).resolve()
        units = robust._training_units(
            seeds=(seed,), stochastic_arms=(settings.arms["C99"], settings.arms["C101"]),
            settings=settings, completed=set(), model_root=root / "strategy_model",
            run_root=root / "strategy_run", comparison_start="2021-01-01",
            comparison_end="2026-03-02", reuse_completed=True,
            benchmark_id=cfg.benchmark_id, checkpoint_cache_root=cfg.checkpoint_cache_root,
        )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "strategy_checkpoint_cache_uses_research_shared_fitting_identity_root_not_seed_silo",
            True, len(units) == 1 and len(units[0]["replay_arms"]) == 2
            and Path(units[0]["checkpoint_cache_root"]) == shared_root,
        )

        # Execution-plan terminology must distinguish reused model evidence
        # from still-pending scientific strategy observations for all six arms.
        status = {
            "preparation_plan": SimpleNamespace(actions=()),
            "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
            "artifact_identities": {},
        }
        fixed_arms, strategy_arms = robust._robustness_arms(settings, cfg)
        ready_contract = {"fingerprint": "synthetic", "resolved_seeds": cfg.resolved_seeds}
        with patch.object(
            robust, "_discover_registered_strategy_only_scientific_reuse", return_value={}
        ), patch.object(
            robust, "_benchmark_parameter_plan_rows", return_value=([], [], {})
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
                missing_plan, _missing_state = robust._render_robustness_execution_plan(
                    cfg=cfg, settings=settings, status=status,
                    fixed_arms=fixed_arms, stochastic_arms=strategy_arms,
                )
        add_check(
            results, "synthetic_breakout_quality", case_id,
            "six_arm_plan_reports_model_reuse_and_only_replays_new_strategy_scientific_units",
            True, reused_state["overall_status"] == "READY"
            and reused_plan.count("REUSE+REPLAY") == 6
            and "0 待PIT／checkpoint確認" in reused_plan
            and "Research模型score REUSE=12" in reused_plan
            and missing_plan.count("TRAIN+REPLAY") == 6
            and "Research模型score REUSE=0" in missing_plan,
        )
