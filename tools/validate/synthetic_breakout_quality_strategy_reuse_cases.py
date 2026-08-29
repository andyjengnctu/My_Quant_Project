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

    from config import strategy_compare as strategy_config
    from services.research import strategy_comparison as score_reuse_module

    reuse_source = (
        project_root / "services" / "research" / "strategy_compare_reuse.py"
    ).read_text(encoding="utf-8")
    orchestration_source = (
        project_root / "services" / "research" / "strategy_comparison.py"
    ).read_text(encoding="utf-8")
    engine_source = (
        project_root / "services" / "research" / "strategy_compare_engine.py"
    ).read_text(encoding="utf-8")

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
        and "BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_CONSTRAINED_OPTIMAL"
        in engine_source,
    )

    forward_score_reuse_base = strategy_config.get_strategy_comparison_settings(
        "forward_oos"
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
