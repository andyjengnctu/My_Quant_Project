from __future__ import annotations

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    LABEL_PASS,
    Path,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    SimpleNamespace,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    _resolve_candidate_quality_ranking,
    bind_checks,
    ast,
    build_file_manifest,
    get_breakout_quality_experiment_profile,
    json,
    np,
    patch,
    pd,
    replace,
    resolve_breakout_quality_rank,
    sort_candidate_rows,
    tempfile,
)

from .source_index import read_source_ast, read_source_text

def validate_breakout_quality_point_in_time_score_builder_contract_case(_base_params):
    """Protect PIT legality, fold identity and score-universe semantics."""
    case_id = "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_BUILDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config import breakout_quality as workflow_config
    from config.breakout_quality import (
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_workflow_settings,
    )
    from filters.breakout_quality.continuous_ranker_data import _validate_group_consistency
    from services.breakout_quality.point_in_time_audit import (
        _build_hs_conditional_mfe_evaluation,
        _build_safety_raw_mfe_evaluation,
    )
    from filters.breakout_quality.ranker_sample_contract import (
        resolve_forward_oos_score_group_ids,
        resolve_forward_oos_target_evaluable_group_ids,
    )
    from tools.filters.breakout_quality.build_point_in_time_scores import (
        _build_fold_periods,
        _checkpoint_matches_rescore_contract,
        _fold_training_contract_is_compatible,
        _forward_checkpoint_import_issues,
        _publish_fold_checkpoint_to_cache,
        _reload_fold_manifest_after_checkpoint_cache_sync,
        _score_output_capability,
        _score_output_columns,
        fold_training_identity,
        _resolve_training_universe_start,
        _stable_fold_id,
    )

    terminal = pd.DataFrame([
        {"ticker": "2330", "date": "2026-03-02", "group_index": 0, "label_eval_end_date": None},
        {"ticker": "2330", "date": "2026-03-02", "group_index": 0, "label_eval_end_date": None},
    ])
    try:
        _validate_group_consistency(terminal)
        terminal_ok = True
    except ValueError:
        terminal_ok = False
    check_true("pit_group_consistency_accepts_all_missing_terminal_label_end", terminal_ok)

    check(
        "pit_training_history_start_is_data_driven",
        pd.Timestamp("2004-09-08"),
        _resolve_training_universe_start(
            SimpleNamespace(summary={"training_universe_start_date": "2004-09-08"}),
            selection_start=pd.Timestamp("2011-01-01"),
        ),
    )
    try:
        _resolve_training_universe_start(
            SimpleNamespace(
                summary={},
                profile=SimpleNamespace(
                    training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                ),
            ),
            selection_start=pd.Timestamp("2011-01-01"),
        )
    except ValueError:
        missing_history_rejected = True
    else:
        missing_history_rejected = False
    check_true("daily_universal_missing_history_start_fails_closed", missing_history_rejected)

    # Forward→Rolling robustness bridge: current Forward artifacts persist sample scope
    # inside experiment_settings (not as a top-level manifest/checkpoint field), and the
    # checkpoint itself does not persist selected_epoch.  The bridge must therefore use
    # canonical manifest/report/hash/split/execution evidence without requiring fields that
    # do not exist in already-trained [5] artifacts.
    execution_payload = {
        "requested_device": "auto",
        "resolved_device": "cuda",
        "mixed_precision_requested": True,
        "mixed_precision_enabled": True,
        "autocast_dtype": "bfloat16",
        "deterministic_algorithms": True,
        "allow_tf32": False,
    }
    profile_payload = {
        "name": "synthetic_daily_profile",
        "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    }
    checkpoint_manifest = {"sha256": "abc", "size_bytes": 123}
    source_contract = {
        "dataset_policy": "synthetic",
        "dataset_storage_schema_version": 1,
        "source_data_inventory": {"sha256": "inventory"},
        "dataset_artifacts": {"groups": {"sha256": "groups"}},
        "target_schema_version": 1,
        "target_contract": {"target_id": "synthetic_target"},
        "target_artifacts": {"raw": {"sha256": "target"}},
        "training_universe_start_date": "2004-09-08",
    }
    fold_contract = {
        "filter_id": "synthetic_quality",
        "model_architecture": "synthetic_arch",
        "experiment_profile": "synthetic_daily_profile",
        "continuous_target_id": "synthetic_target",
        "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
        "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        "model_information_cutoff": "2020-12-31",
        "model_spec": {"name": "synthetic_model"},
        "experiment_settings": profile_payload,
        "seed": 693545351,
        "training_settings": {
            "batch_size": 128,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
        "planned_periods": {
            "validation_start": "2019-01-01",
            "validation_end": "2020-12-31",
            "score_start": "2021-01-01",
        },
        "group_counts": {"inner_train": 1000, "validation": 200, "final_refit": 1200},
        "source_contract": source_contract,
    }
    source_manifest = {
        **{
            key: fold_contract[key]
            for key in (
                "filter_id", "model_architecture", "experiment_profile",
                "continuous_target_id", "training_label_scope",
                "model_information_cutoff", "model_spec", "experiment_settings",
            )
        },
        "selected_epoch": 2,
        "torch_execution": execution_payload,
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
        "model": checkpoint_manifest,
    }
    source_report = {
        "training": {
            "seed": fold_contract["seed"],
            "selected_epoch": 2,
            "batch_size": 128,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
        "standard_model_sop": {"evaluation_mode": "forward_oos"},
        "split_report": {
            "selection_start_date": source_contract["training_universe_start_date"],
            "inner_validation_start_date": "2019-01-01",
            "selection_end_date": "2020-12-31",
            "oos_start_date": "2021-01-01",
            "counts": {"inner_train": 1000, "validation": 200, "selection": 1200},
        },
        "torch_execution": execution_payload,
        "artifacts": {"model": checkpoint_manifest},
    }
    checkpoint = {
        "model_spec": fold_contract["model_spec"],
        "experiment_settings": profile_payload,
        "experiment_profile": fold_contract["experiment_profile"],
        "sequence_length": 300,
        "feature_count": 10,
        "context_count": 5,
    }
    import_bundle = SimpleNamespace(
        feature_bank=np.zeros((2, 300, 10), dtype=np.float32),
        group_context=np.zeros((2, 5), dtype=np.float32),
    )
    execution_plan = SimpleNamespace(as_manifest_payload=lambda: execution_payload)
    import_issues = _forward_checkpoint_import_issues(
        source_manifest=source_manifest,
        source_report=source_report,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        fold_contract=fold_contract,
        bundle=import_bundle,
        execution_plan=execution_plan,
    )
    check_true(
        "forward_robustness_checkpoint_matches_first_rolling_fit_without_nonexistent_checkpoint_fields",
        import_issues == (),
    )
    imported_cache_manifest = {
        **fold_contract,
        "selected_epoch": 2,
        "migration": {
            "kind": "forward_model_fitting_identity_import",
            "training_contract_unchanged": True,
            "source_checkpoint_sha_preserved": True,
        },
    }
    check_true(
        "forward_imported_checkpoint_can_rescore_without_pit_only_checkpoint_fields",
        _checkpoint_matches_rescore_contract(
            checkpoint=checkpoint,
            manifest=imported_cache_manifest,
            fold_contract=fold_contract,
            bundle=import_bundle,
        ),
    )
    stale_fold_contract = {**fold_contract, "model_information_cutoff": "2021-12-31"}
    stale_issues = _forward_checkpoint_import_issues(
        source_manifest=source_manifest,
        source_report=source_report,
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest,
        fold_contract=stale_fold_contract,
        bundle=import_bundle,
        execution_plan=execution_plan,
    )
    check_true(
        "forward_robustness_checkpoint_never_crosses_different_rolling_cutoff",
        "manifest model_information_cutoff mismatch" in stale_issues,
    )

    event_profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    bundle = SimpleNamespace(
        group_table=pd.DataFrame([
            {"ticker": "A", "date": "2021-01-04", "group_index": 0, "label": 1},
            {"ticker": "B", "date": "2021-01-04", "group_index": 1, "label": 0},
            {"ticker": "C", "date": "2021-01-05", "group_index": 2, "label": -1},
            {"ticker": "D", "date": "2020-12-31", "group_index": 3, "label": 1},
        ]),
        target_valid=np.array([True, False, True, True], dtype=bool),
        outer_policy={"oos_start_date": "2021-01-01", "effective_oos_end_date": "2021-12-31"},
        profile=event_profile,
    )
    check(
        "forward_oos_score_universe_is_future_target_independent",
        ([0, 1, 2], [0]),
        (
            resolve_forward_oos_score_group_ids(bundle).tolist(),
            resolve_forward_oos_target_evaluable_group_ids(bundle).tolist(),
        ),
    )

    settings = get_breakout_quality_workflow_settings()
    binary = workflow_config.get_breakout_quality_workflow_settings(
        experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
    )
    continuous = workflow_config.get_breakout_quality_workflow_settings(
        experiment_profile=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    daily = workflow_config.get_breakout_quality_workflow_settings(
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    check_true(
        "workflow_profiles_keep_binary_continuous_and_daily_pit_semantics",
        binary.training_objective == "binary_classification"
        and continuous.strategy_score_source == "selection_point_in_time"
        and daily.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        and daily.supports_point_in_time_scores,
    )

    oos = _build_fold_periods(
        pd.Timestamp("2021-01-01"), pd.Timestamp("2026-03-02"),
        fold_months=12, fold_anchor=pd.Timestamp("2021-01-01"), single_score_block=True,
    )
    rolling = _build_fold_periods(
        pd.Timestamp("2021-01-01"), pd.Timestamp("2026-03-02"),
        fold_months=12, single_score_block=False,
    )
    check(
        "pit_oos_is_single_block_and_rolling_is_annual_with_partial_tail",
        (1, 6, "2021-01-01", "2026-03-02"),
        (
            len(oos), len(rolling),
            str(rolling[0]["score_start"].date()),
            str(rolling[-1]["score_end"].date()),
        ),
    )
    check(
        "pit_fold_identity_is_date_stable",
        "fold_20140101_20141231",
        _stable_fold_id(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")),
    )

    expected_contract = {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        "continuous_target_id": "daily_opportunity_no_time_r_v1",
        "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
        "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        "seed": 42,
        "model_information_cutoff": "2020-12-31",
        "model_spec": {"architecture": "inception_time_v1"},
        "experiment_settings": {"profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE},
        "training_settings": {"objective": "daily_pairwise_ranking"},
        "source_contract": {"training_universe_start_date": "2004-09-08"},
        "lookahead_contract": {"future_target_in_scoring": False},
        "planned_periods": {
            "validation_start": "2019-01-01",
            "validation_end": "2020-12-31",
            "score_start": "2021-01-01",
            "score_end": "2026-03-02",
            "history_start": "2004-09-08",
        },
        "observed_periods": {phase: phase for phase in ("inner_train", "validation", "final_refit")},
        "group_counts": {phase: 10 for phase in ("inner_train", "validation", "final_refit")},
        "event_row_counts": {phase: 10 for phase in ("inner_train", "validation", "final_refit")},
    }
    same_contract = {**expected_contract, "planned_periods": {**expected_contract["planned_periods"], "score_end": "2021-12-31"}}
    changed_contract = dict(expected_contract, model_information_cutoff="2021-12-31")
    check_true(
        "pit_fold_reuse_ignores_score_horizon_but_rejects_training_identity_change",
        _fold_training_contract_is_compatible(same_contract, expected_contract=expected_contract)
        and not _fold_training_contract_is_compatible(changed_contract, expected_contract=expected_contract),
    )

    a1_profile = workflow_config.get_breakout_quality_experiment_profile(
        workflow_config.DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    a2_profile = workflow_config.get_breakout_quality_experiment_profile(
        workflow_config.DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    expected_raw_duo_columns = {
        "primary": "breakout_quality_score",
        "raw_safety": "raw_safety_score",
        "raw_mfe": "raw_mfe_score",
    }
    from config.breakout_quality_runtime import (
        get_continuous_ranker_persisted_score_columns,
        get_continuous_ranker_score_output_columns,
        get_continuous_ranker_training_policy,
    )
    from filters.breakout_quality.ranking_score_store import PIT_OPTIONAL_SCORE_COLUMNS
    from services.breakout_quality.point_in_time_scores import OPTIONAL_SCORE_COLUMNS

    a1_training_policy = get_continuous_ranker_training_policy(a1_profile.training_objective)
    a2_training_policy = get_continuous_ranker_training_policy(a2_profile.training_objective)
    a1_output_policy = a1_training_policy.score_output_policy
    a2_output_policy = a2_training_policy.score_output_policy
    pit_score_source = read_source_text("services/breakout_quality/point_in_time_scores.py")
    check_true(
        "pit_runtime_training_policy_owns_sidecars_independent_of_model_topology",
        a1_output_policy is not None
        and a1_output_policy == a2_output_policy
        and a1_output_policy.output_head == "both"
        and a1_output_policy.manifest_columns() == expected_raw_duo_columns
        and _score_output_columns(SimpleNamespace(profile=a1_profile)) == expected_raw_duo_columns
        and _score_output_columns(SimpleNamespace(profile=a2_profile)) == expected_raw_duo_columns,
    )
    check_true(
        "pit_score_output_capability_is_single_runtime_owner_for_all_consumers",
        get_continuous_ranker_score_output_columns(a1_profile.training_objective)
        == expected_raw_duo_columns
        and get_continuous_ranker_score_output_columns(a2_profile.training_objective)
        == expected_raw_duo_columns
        and "get_continuous_ranker_score_output_columns" in read_source_text(
            "filters/breakout_quality/ranking_score_store.py"
        )
        and "get_continuous_ranker_score_output_columns" in read_source_text(
            "services/breakout_quality/point_in_time_audit.py"
        ),
        note=(
            "PIT score readiness and audit readiness must consume the same declarative "
            "score-output capability; no objective-specific sidecar matrix is allowed."
        ),
    )
    from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
    from filters.breakout_quality.ranking_score_store import (
        _validate_score_eligibility_contract,
        _validate_score_output_capability_contract,
    )

    a1_manifest_base = {
        "score_eligibility_contract": build_score_eligibility_contract(a1_profile),
    }
    current_a1_manifest = {**a1_manifest_base, "score_columns": expected_raw_duo_columns}
    _validate_score_eligibility_contract(current_a1_manifest, profile=a1_profile)
    _validate_score_output_capability_contract(current_a1_manifest, profile=a1_profile)
    eligibility_only_manifest_accepted = True
    try:
        _validate_score_eligibility_contract(a1_manifest_base, profile=a1_profile)
    except ValueError:
        eligibility_only_manifest_accepted = False
    output_capability_rejects_eligibility_only_manifest = False
    try:
        _validate_score_output_capability_contract(a1_manifest_base, profile=a1_profile)
    except ValueError as exc:
        output_capability_rejects_eligibility_only_manifest = (
            "score-output capability" in str(exc)
        )
    check_true(
        "pit_row_eligibility_and_output_capability_contracts_are_orthogonal",
        eligibility_only_manifest_accepted
        and output_capability_rejects_eligibility_only_manifest,
        note=(
            "Prediction-time row eligibility must remain future-independent and must not be "
            "redefined by multi-head sidecar completeness; the full PIT loader validates both "
            "contracts independently."
        ),
    )
    stale_a1_rejected = False
    try:
        _validate_score_output_capability_contract(
            {**a1_manifest_base, "score_columns": {"primary": "breakout_quality_score"}},
            profile=a1_profile,
        )
    except ValueError as exc:
        stale_a1_rejected = "score-output capability" in str(exc)
    check_true(
        "pit_reader_rejects_stale_multi_head_score_capability_before_audit_refresh",
        stale_a1_rejected,
        note=(
            "A stale AK/AO-family PIT score must enter checkpoint-rescore, not audit-only refresh; "
            "otherwise missing model-output sidecars can never be reconstructed."
        ),
    )
    check_true(
        "pit_score_output_consumer_has_no_local_target_builder_capability_registry",
        "_PIT_SCORE_OUTPUT_CAPABILITIES" not in pit_score_source
        and "CONTINUOUS_RANKER_TARGET_BUILDER_" not in pit_score_source,
    )
    registered_optional_score_columns = {
        str(column)
        for training_objective in workflow_config.CONTINUOUS_RANKER_TRAINING_OBJECTIVES
        for training_policy in (
            get_continuous_ranker_training_policy(training_objective),
        )
        if training_policy.score_output_policy is not None
        for _head_name, column in training_policy.score_output_policy.persisted_columns
        if str(column) != "breakout_quality_score"
    }
    runtime_optional_score_columns = get_continuous_ranker_persisted_score_columns()
    ranking_store_source = read_source_text("filters/breakout_quality/ranking_score_store.py")
    check_true(
        "pit_optional_sidecar_column_union_is_runtime_owned_for_producer_and_reader",
        set(runtime_optional_score_columns) == registered_optional_score_columns
        and len(runtime_optional_score_columns) == len(set(runtime_optional_score_columns))
        and OPTIONAL_SCORE_COLUMNS == runtime_optional_score_columns
        and PIT_OPTIONAL_SCORE_COLUMNS == runtime_optional_score_columns
        and "PIT_OPTIONAL_SCORE_COLUMNS = (" not in ranking_store_source
        and "OPTIONAL_SCORE_COLUMNS = (" not in pit_score_source,
    )

    score_output_only_change = {
        **expected_contract,
        "score_output_contract": {
            "primary": "breakout_quality_score",
            "conditional_mfe": "breakout_quality_score",
            "raw_safety": "raw_safety_score",
        },
    }
    raw_duo_score_output_change = {
        **expected_contract,
        "score_output_contract": expected_raw_duo_columns,
    }
    check_true(
        "pit_score_output_contract_change_reuses_same_fitting_identity_checkpoint",
        _fold_training_contract_is_compatible(
            score_output_only_change, expected_contract=expected_contract
        )
        and _fold_training_contract_is_compatible(
            raw_duo_score_output_change, expected_contract=expected_contract
        )
        and fold_training_identity(score_output_only_change)
        == fold_training_identity(expected_contract)
        == fold_training_identity(raw_duo_score_output_change),
    )

    group_count = 20
    dates = pd.to_datetime(["2026-01-05"] * 10 + ["2026-01-06"] * 10)
    favorable = np.linspace(0.1, 2.0, group_count, dtype=np.float32)
    adverse = np.linspace(1.0, 0.1, group_count, dtype=np.float32)
    group_table = pd.DataFrame({
        "group_index": np.arange(group_count, dtype=np.int64),
        "date": dates,
        "label": np.asarray([0, 1] * (group_count // 2), dtype=np.int64),
        "target_favorable_r": favorable,
        "target_adverse_r": adverse,
    })
    audit_bundle = SimpleNamespace(
        group_table=group_table,
        raw_target=favorable.copy(),
        target_valid=np.ones(group_count, dtype=bool),
    )
    audit_frame = pd.DataFrame({
        "group_index": np.arange(group_count, dtype=np.int64),
        "target_available": np.ones(group_count, dtype=bool),
        "raw_safety_score": np.linspace(0.95, 0.05, group_count),
        "raw_mfe_score": np.linspace(0.05, 0.95, group_count),
    })
    raw_eval = _build_safety_raw_mfe_evaluation(
        audit_bundle, audit_frame, np.arange(0, group_count, 2, dtype=np.int64)
    )
    check_true(
        "rolling_audit_consumes_persisted_raw_head_sidecars_into_model_specific_evidence",
        set(raw_eval) == {"oos", "breakout_candidate_oos"}
        and int((raw_eval["oos"].get("raw_safety") or {}).get("group_count", 0)) == group_count
        and int((raw_eval["oos"].get("raw_mfe") or {}).get("group_count", 0)) == group_count
        and bool(raw_eval["oos"].get("model_gate")),
    )

    hs_frame = audit_frame.drop(columns=["raw_mfe_score"]).copy()
    hs_frame["date"] = dates
    hs_frame["breakout_quality_score"] = np.linspace(0.05, 0.95, group_count)
    hs_eval = _build_hs_conditional_mfe_evaluation(
        audit_bundle,
        hs_frame,
        np.arange(0, group_count, 2, dtype=np.int64),
        experiment_profile=(
            DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
        ),
    )
    check_true(
        "rolling_audit_rebuilds_hs_conditional_extension_from_persisted_primary_and_safety_sidecar",
        set(hs_eval) == {"oos", "breakout_candidate_oos"}
        and bool((hs_eval["oos"].get("raw_safety") or {}))
        and bool((hs_eval["oos"].get("hs_qualification") or {}))
        and bool((hs_eval["oos"].get("conditional_mfe_true_hs") or {}))
        and bool((hs_eval["oos"].get("lexicographic_model_gate") or {})),
    )

    # Reproduce the C71 score-only upgrade failure: legacy code refreshed the
    # embedded fold_contract inside model.pt, changing its SHA while fitting identity
    # stayed unchanged.  The fitting cache is canonical and must restore its exact
    # checkpoint bytes instead of treating this resumable score-only migration as a
    # second fitted model.
    with tempfile.TemporaryDirectory(prefix="pit_score_only_cache_repair_") as temp_dir:
        root = Path(temp_dir)
        fold_dir = root / "fold"
        cache_root = root / "cache"
        fold_dir.mkdir(parents=True)
        cache_entry = cache_root / fold_training_identity(expected_contract)
        cache_entry.mkdir(parents=True)
        cached_model = cache_entry / "model.pt"
        local_model = fold_dir / "model.pt"
        cached_model.write_bytes(b"canonical-fitting-checkpoint")
        local_model.write_bytes(b"score-only-metadata-rewritten-checkpoint")
        cached_checkpoint = build_file_manifest(cached_model)
        local_checkpoint = build_file_manifest(local_model)
        cached_manifest = {
            **expected_contract,
            "artifacts": {"checkpoint": cached_checkpoint},
        }
        local_manifest = {
            **score_output_only_change,
            "migration": {
                "kind": "expanded_daily_score_universe_checkpoint_reuse",
                "training_contract_unchanged": True,
            },
            "artifacts": {"checkpoint": local_checkpoint},
        }
        (cache_entry / "manifest.json").write_text(
            json.dumps(cached_manifest), encoding="utf-8"
        )
        (fold_dir / "manifest.json").write_text(
            json.dumps(local_manifest), encoding="utf-8"
        )
        publish_result = _publish_fold_checkpoint_to_cache(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=score_output_only_change
        )
        repaired_manifest = json.loads(
            (fold_dir / "manifest.json").read_text(encoding="utf-8")
        )
        repaired_checkpoint = build_file_manifest(local_model)
    check_true(
        "pit_score_only_rescore_checkpoint_sha_conflict_restores_canonical_fitting_cache",
        bool(
            publish_result
            and publish_result.get("reused_existing_cache") is True
            and publish_result.get("restored_source_fold_checkpoint") is True
            and repaired_checkpoint == cached_checkpoint
            and repaired_manifest["artifacts"]["checkpoint"] == cached_checkpoint
            and repaired_manifest["migration"].get("canonical_checkpoint_cache_restore") is True
            and repaired_manifest["migration"].get("source_checkpoint_sha_preserved") is True
        ),
    )

    # The publisher may rewrite the fold manifest during canonical cache repair.  The
    # aggregate builder must reload that post-sync manifest instead of retaining the
    # pre-repair in-memory SHA that triggered the 2026-08-26 Rolling PIT audit failure.
    with tempfile.TemporaryDirectory(prefix="pit_post_cache_manifest_sync_") as temp_dir:
        root = Path(temp_dir)
        fold_dir = root / "fold"
        cache_root = root / "cache"
        fold_dir.mkdir(parents=True)
        cache_entry = cache_root / fold_training_identity(expected_contract)
        cache_entry.mkdir(parents=True)
        cached_model = cache_entry / "model.pt"
        local_model = fold_dir / "model.pt"
        score_path = fold_dir / "scores.csv"
        validation_score_path = fold_dir / "validation_scores.csv"
        cached_model.write_bytes(b"canonical-fitting-checkpoint")
        local_model.write_bytes(b"metadata-only-drifted-local-checkpoint")
        score_path.write_text("ticker,date\n2330,2021-01-04\n", encoding="utf-8")
        validation_score_path.write_text("ticker,date\n2330,2020-12-31\n", encoding="utf-8")
        cached_checkpoint = build_file_manifest(cached_model)
        stale_local_checkpoint = build_file_manifest(local_model)
        cached_manifest = {
            **expected_contract,
            "artifacts": {"checkpoint": cached_checkpoint},
        }
        stale_in_memory_manifest = {
            **score_output_only_change,
            "migration": {
                "kind": "expanded_daily_score_universe_checkpoint_reuse",
                "training_contract_unchanged": True,
            },
            "artifacts": {
                "checkpoint": stale_local_checkpoint,
                "scores": build_file_manifest(score_path),
                "validation_scores": build_file_manifest(validation_score_path),
            },
        }
        (cache_entry / "manifest.json").write_text(
            json.dumps(cached_manifest), encoding="utf-8"
        )
        (fold_dir / "manifest.json").write_text(
            json.dumps(stale_in_memory_manifest), encoding="utf-8"
        )
        _publish_fold_checkpoint_to_cache(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=score_output_only_change
        )
        synced_manifest = _reload_fold_manifest_after_checkpoint_cache_sync(
            fold_dir=fold_dir, fold_contract=score_output_only_change
        )
        aggregate_checkpoint_record = dict(synced_manifest["artifacts"]["checkpoint"])
        actual_checkpoint_record = build_file_manifest(local_model)
    check_true(
        "pit_aggregate_reloads_post_cache_repair_fold_manifest_checkpoint_sha",
        bool(
            stale_in_memory_manifest["artifacts"]["checkpoint"] != actual_checkpoint_record
            and aggregate_checkpoint_record == actual_checkpoint_record
            and synced_manifest["migration"].get("canonical_checkpoint_cache_restore") is True
        ),
    )

    with tempfile.TemporaryDirectory(prefix="pit_true_checkpoint_conflict_") as temp_dir:
        root = Path(temp_dir)
        fold_dir = root / "fold"
        cache_root = root / "cache"
        fold_dir.mkdir(parents=True)
        cache_entry = cache_root / fold_training_identity(expected_contract)
        cache_entry.mkdir(parents=True)
        cached_model = cache_entry / "model.pt"
        local_model = fold_dir / "model.pt"
        cached_model.write_bytes(b"canonical-fitting-checkpoint")
        local_model.write_bytes(b"genuinely-different-fitting-checkpoint")
        cached_manifest = {
            **expected_contract,
            "artifacts": {"checkpoint": build_file_manifest(cached_model)},
        }
        local_manifest = {
            **expected_contract,
            "artifacts": {"checkpoint": build_file_manifest(local_model)},
        }
        (cache_entry / "manifest.json").write_text(json.dumps(cached_manifest), encoding="utf-8")
        (fold_dir / "manifest.json").write_text(json.dumps(local_manifest), encoding="utf-8")
        try:
            _publish_fold_checkpoint_to_cache(
                fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract
            )
        except RuntimeError as exc:
            true_conflict_rejected = "同一fitting identity產生不同checkpoint" in str(exc)
        else:
            true_conflict_rejected = False
    check_true(
        "pit_true_same_identity_different_checkpoint_still_fails_closed",
        true_conflict_rejected,
    )

    pit_source = read_source_text("services/breakout_quality/point_in_time_scores.py")
    rescore_start = pit_source.index("def _rescore_fold_from_compatible_checkpoint(")
    rescore_end = pit_source.index("def _rescore_fold_from_fitting_checkpoint(", rescore_start)
    rescore_source = pit_source[rescore_start:rescore_end]
    check_true(
        "pit_score_only_rescore_does_not_rewrite_fitted_checkpoint_bytes",
        "torch_module.save(" not in rescore_source
        and "source_checkpoint_sha_preserved" in rescore_source,
    )

    loop_anchor = pit_source.index("for fold_index, (fold, ids) in enumerate")
    publish_call = pit_source.index("_publish_fold_checkpoint_to_cache(", loop_anchor)
    post_sync_reload = pit_source.index("_reload_fold_manifest_after_checkpoint_cache_sync(", publish_call)
    aggregate_append = pit_source.index("fold_manifests.append(manifest)", post_sync_reload)
    check_true(
        "pit_cache_sync_reloads_fold_manifest_before_top_level_aggregate_append",
        publish_call < post_sync_reload < aggregate_append,
    )

    reuse_branch_start = pit_source.index("if reused is not None:", loop_anchor)
    train_branch_start = pit_source.index("else:\n            if bool(args.checkpoint_only):", reuse_branch_start)
    reuse_branch_source = pit_source[reuse_branch_start:train_branch_start]
    check_true(
        "pit_compact_console_reports_every_reused_fold_with_source_and_progress",
        'fold_progress.print_line(' in reuse_branch_source
        and 'paint("REUSE", "green"' in reuse_branch_source
        and 'f"PIT fold {fold_index}/{len(folds)} | "' in reuse_branch_source
        and 'reuse_source = "Rolling fold artifact"' in reuse_branch_source
        and 'reuse_source = "Rolling fold checkpoint"' in reuse_branch_source
        and 'reuse_source = "shared fitting cache"' in reuse_branch_source
        and 'Forward OOS fitting checkpoint → shared cache' in reuse_branch_source,
    )

    oos_2023 = {
        **expected_contract,
        "model_information_cutoff": "2022-12-31",
        "planned_periods": {
            **expected_contract["planned_periods"],
            "validation_end": "2022-12-31",
            "score_start": "2023-01-01",
            "score_end": "2026-03-02",
        },
    }
    rolling_2023 = {
        **oos_2023,
        "planned_periods": {**oos_2023["planned_periods"], "score_end": "2023-12-31"},
    }
    check_true(
        "pit_fitting_identity_cache_is_date_generic_not_initial_year_special_case",
        fold_training_identity(oos_2023) == fold_training_identity(rolling_2023)
        and fold_training_identity(oos_2023) != fold_training_identity(expected_contract),
    )

    from filters.breakout_quality.ranking_score_store import (
        load_selection_point_in_time_score_table_from_path,
        lookup_selection_point_in_time_candidate_score,
    )
    with tempfile.TemporaryDirectory(prefix="mr13p_dual_head_pit_") as temp_dir:
        score_path = Path(temp_dir) / "selection_point_in_time_scores.csv"
        manifest_path = Path(temp_dir) / "selection_point_in_time_manifest.json"
        pd.DataFrame([
            {
                "ticker": "2330",
                "date": "2021-01-04",
                "group_index": 1,
                "breakout_quality_score": 0.81,
                "primary_mfe_score": 0.81,
                "conditional_safety_score": 0.73,
                "fold_id": "fold_20210101_20211231",
                "model_information_cutoff": "2020-12-31",
            }
        ]).to_csv(score_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(json.dumps({
            "score_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "coverage": {"scored_group_count": 1},
        }), encoding="utf-8")
        dual_table = load_selection_point_in_time_score_table_from_path(
            str(score_path), manifest_path=str(manifest_path)
        )
        conditional_lookup = lookup_selection_point_in_time_candidate_score(
            project_root=str(Path(__file__).resolve().parents[2]),
            ticker="2330",
            signal_date="2021-01-04",
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_conditional_mfe_safety_v1",
            experiment_profile="daily_universal_conditional_mfe_safety_full_list_ndcg_pairwise",
            score_path_override=str(score_path),
            manifest_path_override=str(manifest_path),
            score_column="conditional_safety_score",
        )
    check_true(
        "mr13p_dual_head_pit_retains_conditional_score_and_lookup_selects_it_without_second_source",
        "conditional_safety_score" in dual_table.columns
        and abs(float(conditional_lookup["score"]) - 0.73) < 1e-12
        and bool(conditional_lookup["available"]),
    )


    with tempfile.TemporaryDirectory(prefix="mr13r_dual_head_pit_") as temp_dir:
        score_path = Path(temp_dir) / "selection_point_in_time_scores.csv"
        manifest_path = Path(temp_dir) / "selection_point_in_time_manifest.json"
        pd.DataFrame([
            {
                "ticker": "2330",
                "date": "2021-01-04",
                "group_index": 1,
                "breakout_quality_score": 0.91,
                "raw_safety_score": 0.88,
                "fold_id": "fold_20210101_20211231",
                "model_information_cutoff": "2020-12-31",
            }
        ]).to_csv(score_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(json.dumps({
            "score_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "coverage": {"scored_group_count": 1},
            "score_columns": {
                "primary": "breakout_quality_score",
                "conditional_mfe": "breakout_quality_score",
                "raw_safety": "raw_safety_score",
            },
        }), encoding="utf-8")
        reverse_dual_table = load_selection_point_in_time_score_table_from_path(
            str(score_path), manifest_path=str(manifest_path)
        )
        raw_safety_lookup = lookup_selection_point_in_time_candidate_score(
            project_root=str(Path(__file__).resolve().parents[2]),
            ticker="2330",
            signal_date="2021-01-04",
            filter_id="breakout_quality_v1",
            model_architecture="inception_time_safety_conditional_mfe_v1",
            experiment_profile="daily_universal_safety_conditional_mfe_duo_head_full_list_ndcg_pairwise",
            score_path_override=str(score_path),
            manifest_path_override=str(manifest_path),
            score_column="raw_safety_score",
        )
    check_true(
        "mr13r_pit_retains_raw_safety_score_and_lookup_selects_same_source_aux_head",
        "raw_safety_score" in reverse_dual_table.columns
        and abs(float(raw_safety_lookup["score"]) - 0.88) < 1e-12
        and bool(raw_safety_lookup["available"]),
    )

    from filters.breakout_quality.runtime import (
        breakout_quality_ranking_source_context,
        resolve_breakout_quality_candidate_rank,
    )
    with tempfile.TemporaryDirectory(prefix="mr13z_joint_min_pit_") as temp_dir:
        score_path = Path(temp_dir) / "selection_point_in_time_scores.csv"
        manifest_path = Path(temp_dir) / "selection_point_in_time_manifest.json"
        pd.DataFrame([
            {
                "ticker": "2330",
                "date": "2021-01-04",
                "group_index": 1,
                "breakout_quality_score": 0.31,
                "raw_safety_score": 0.82,
                "raw_mfe_score": 0.31,
                "joint_min_score": 0.67,
                "fold_id": "fold_20210101_20211231",
                "model_information_cutoff": "2020-12-31",
            }
        ]).to_csv(score_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(json.dumps({
            "score_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "coverage": {"scored_group_count": 1},
            "score_columns": {
                "primary": "breakout_quality_score",
                "raw_safety": "raw_safety_score",
                "raw_mfe": "raw_mfe_score",
                "joint_min": "joint_min_score",
            },
        }), encoding="utf-8")
        joint_table = load_selection_point_in_time_score_table_from_path(
            str(score_path), manifest_path=str(manifest_path)
        )
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1",
            experiment_profile="daily_universal_safety_raw_mfe_joint_min_patch_transformer_attn_pool_mlp_head_full_list_ndcg_pairwise",
            ranking_options={"primary_score_column": "joint_min_score"},
            score_path_override=str(score_path),
            score_manifest_path_override=str(manifest_path),
        ):
            joint_runtime = resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date="2021-01-05",
                information_date="2021-01-04",
                high_len=275,
                project_root=str(Path(__file__).resolve().parents[2]),
            )
    check_true(
        "mr13z_pit_preserves_all_three_heads_and_runtime_primary_explicitly_consumes_joint_min",
        {"raw_safety_score", "raw_mfe_score", "joint_min_score"}.issubset(joint_table.columns)
        and abs(float(joint_table.iloc[0]["breakout_quality_score"]) - 0.31) < 1e-12
        and abs(float(joint_runtime["score"]) - 0.67) < 1e-12
        and bool(joint_runtime["available"]),
    )

    audit_source = read_source_text("services/breakout_quality/point_in_time_audit.py")
    ranking_store_source = read_source_text("filters/breakout_quality/ranking_score_store.py")
    pit_manifest_write = pit_source.index("write_json(manifest_path, manifest)")
    pit_cache_clear = pit_source.index(
        "clear_selection_point_in_time_ranking_contract_cache()", pit_manifest_write
    )
    audit_json_write = audit_source.index("write_json(output_json, payload)")
    audit_cache_clear = audit_source.index(
        "clear_selection_point_in_time_ranking_contract_cache()", audit_json_write
    )
    check_true(
        "pit_score_and_audit_producers_invalidate_cached_ranking_contract_after_artifact_rewrite",
        "def clear_selection_point_in_time_ranking_contract_cache()" in ranking_store_source
        and pit_manifest_write < pit_cache_clear
        and audit_json_write < audit_cache_clear,
    )
    check_true(
        "rolling_standard_sop_validation_sidecar_reader_ignores_non_sop_ticker_dtype",
        'usecols=sorted(required)' in audit_source
        and 'dtype={"fold_id": "string"}' in audit_source,
    )

    summary.update({
        "profile": settings.experiment_profile,
        "oos_fold_count": len(oos),
        "rolling_fold_count": len(rolling),
    })
    return results, summary

def validate_breakout_quality_selection_point_in_time_score_sort_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SELECTION_POINT_IN_TIME_SCORE_SORT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from core.buy_sort import (
        BUY_LIMIT_OVERAGE_SORT_METHOD,
        sort_candidate_rows,
    )
    from core.extended_signals import (
        attach_breakout_quality_rank,
        resolve_breakout_quality_rank,
    )
    from core.portfolio_candidates import _make_candidate_row
    from filters.breakout_quality.ranking_score_store import (
        _validate_audit_source_artifact,
        _validate_score_eligibility_contract,
        derive_point_in_time_model_validation_gate,
    )
    from filters.breakout_quality.ranker_sample_contract import (
        build_score_eligibility_contract,
    )
    from filters.breakout_quality.runtime import (
        breakout_quality_ranking_source_context,
        get_breakout_quality_ranking_source_context,
        resolve_breakout_quality_candidate_rank,
    )
    from services.research.strategy_compare_engine import (
        _strategy_selection_diagnostics,
    )

    ranking_rows = [
        {"ticker": "B", "use_breakout_quality_ranking": True,
         "breakout_quality_score": None, "breakout_quality_rank": {"available": False},
         "sort_value": 0.10, "proj_cost": 100.0},
        {"ticker": "C", "use_breakout_quality_ranking": True,
         "breakout_quality_score": 0.80, "breakout_quality_rank": {"available": True},
         "sort_value": 0.50, "proj_cost": 100.0},
        {"ticker": "A", "use_breakout_quality_ranking": True,
         "breakout_quality_score": 0.80, "breakout_quality_rank": {"available": True},
         "sort_value": 0.20, "proj_cost": 100.0},
        {"ticker": "D", "use_breakout_quality_ranking": True,
         "breakout_quality_score": None, "breakout_quality_rank": {"available": False},
         "sort_value": 0.05, "proj_cost": 100.0},
    ]
    sort_candidate_rows(ranking_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
    check(
        "point_in_time_score_sort_desc_tie_and_missing_fallback_contract",
        ["A", "C", "D", "B"],
        [row["ticker"] for row in ranking_rows],
    )

    unavailable_rank = {
        "score": None,
        "available": False,
        "unavailable_reason": "missing_ticker_date_score",
        "score_date": "2014-01-01",
        "score_source": "selection_point_in_time",
        "shared_group_score": True,
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
    }
    signal_state = {}
    attach_breakout_quality_rank(signal_state, unavailable_rank)
    inherited_unavailable_rank = resolve_breakout_quality_rank(signal_state)
    ranking_params = replace(
        _base_params,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            side_effect=AssertionError("延續候選不得重新查詢原事件Score"),
        ):
            resolved_inherited_unavailable_rank = _resolve_candidate_quality_ranking(
                params=ranking_params,
                ticker="0056",
                signal_date=pd.Timestamp("2014-01-01"),
                signal_state=signal_state,
            )
    unavailable_candidate = _make_candidate_row(
        buy_sort_method=BUY_LIMIT_OVERAGE_SORT_METHOD,
        ticker="0056",
        candidate_type="normal",
        est_limit_px=100.0,
        ev=0.0,
        y_atr=2.0,
        t_pos=1,
        y_pos=0,
        est_qty=0,
        win_rate=0.0,
        trade_count=0,
        asset_growth_pct=0.0,
        est_init_sl=90.0,
        est_init_trail=95.0,
        est_target_price=120.0,
        entry_atr=2.0,
        is_orderable=True,
        params=_base_params,
        trade_date=pd.Timestamp("2014-01-02"),
        signal_date=pd.Timestamp("2014-01-01"),
        prev_close=99.0,
        quality_rank=unavailable_rank,
    )
    check(
        "point_in_time_missing_score_is_preserved_for_continuation_and_candidate_fallback",
        (
                    False,
                    None,
                    "missing_ticker_date_score",
                    "selection_point_in_time",
                    "selection_point_in_time",
                    False,
                    None,
                ),
        (
                    inherited_unavailable_rank["available"],
                    inherited_unavailable_rank["score"],
                    inherited_unavailable_rank["unavailable_reason"],
                    inherited_unavailable_rank["score_source"],
                    resolved_inherited_unavailable_rank["score_source"],
                    unavailable_candidate["breakout_quality_rank"]["available"],
                    unavailable_candidate["breakout_quality_score"],
                ),
    )

    default_source = get_breakout_quality_ranking_source_context().score_source
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
    ):
        inside_source = get_breakout_quality_ranking_source_context().score_source
    restored_source = get_breakout_quality_ranking_source_context().score_source
    check(
        "point_in_time_ranking_source_context_is_scoped_and_restored",
        ("canonical_runtime", "selection_point_in_time", "canonical_runtime"),
        (default_source, inside_source, restored_source),
    )

    lookup_dates = []

    def _synthetic_pit_lookup(**kwargs):
        lookup_dates.append(pd.Timestamp(kwargs["signal_date"]).strftime("%Y-%m-%d"))
        return {
            "score": 0.7,
            "available": True,
            "score_date": pd.Timestamp(kwargs["signal_date"]).strftime("%Y-%m-%d"),
            "score_source": "selection_point_in_time",
        }

    with patch(
        "filters.breakout_quality.runtime.lookup_selection_point_in_time_candidate_score",
        side_effect=_synthetic_pit_lookup,
    ):
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_v1",
            experiment_profile=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
        ):
            resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"),
                high_len=275,
            )
        with breakout_quality_ranking_source_context(
            score_source="selection_point_in_time",
            model_architecture="inception_time_v1",
            experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        ):
            resolve_breakout_quality_candidate_rank(
                ticker="2330",
                signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"),
                high_len=275,
            )
    check(
        "pit_runtime_score_date_is_event_anchor_for_mr12b_and_latest_information_date_for_mr13a",
        ["2020-01-02", "2020-01-09"],
        lookup_dates,
    )

    inherited_daily_state = {}
    attach_breakout_quality_rank(
        inherited_daily_state,
        {
            "score": 0.4,
            "available": True,
            "score_date": "2020-01-02",
            "score_source": "selection_point_in_time",
            "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            "model_architecture": "inception_time_v1",
            "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        },
    )
    with breakout_quality_ranking_source_context(
        score_source="selection_point_in_time",
        model_architecture="inception_time_v1",
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            return_value={
                "score": 0.9,
                "available": True,
                "score_date": "2020-01-09",
                "score_source": "selection_point_in_time",
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                "model_architecture": "inception_time_v1",
                "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
            },
        ) as refreshed_lookup:
            refreshed_rank = _resolve_candidate_quality_ranking(
                params=ranking_params,
                ticker="2330",
                signal_date=pd.Timestamp("2020-01-02"),
                information_date=pd.Timestamp("2020-01-09"),
                signal_state=inherited_daily_state,
            )
    check(
        "daily_pit_continuation_refreshes_score_instead_of_reusing_breakout_day_rank",
        (1, 0.9, "2020-01-09"),
        (refreshed_lookup.call_count, refreshed_rank["score"], refreshed_rank["score_date"]),
    )

    daily_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    current_daily_contract = build_score_eligibility_contract(daily_profile)
    current_daily_accepted = True
    try:
        _validate_score_eligibility_contract(
            {"score_eligibility_contract": current_daily_contract},
            profile=daily_profile,
        )
    except ValueError:
        current_daily_accepted = False
    legacy_daily_rejected = False
    try:
        _validate_score_eligibility_contract({}, profile=daily_profile)
    except ValueError as exc:
        legacy_daily_rejected = "重新建立PIT Scores" in str(exc)
    check(
        "strategy_loader_requires_future_independent_daily_pit_score_eligibility_contract",
        (True, True),
        (current_daily_accepted, legacy_daily_rejected),
    )

    with tempfile.TemporaryDirectory() as artifact_dir:
        source_path = Path(artifact_dir) / "selection_point_in_time_scores.csv"
        source_path.write_text("ticker,date,score\n2330,2020-01-01,0.8\n", encoding="utf-8")
        source_record = {"path": str(source_path), **build_file_manifest(source_path)}
        exact_binding_accepted = True
        try:
            _validate_audit_source_artifact(
                source_record,
                expected_path=source_path,
                label="synthetic PIT Scores",
                project_root=Path(artifact_dir),
            )
        except ValueError:
            exact_binding_accepted = False
        source_path.write_text("ticker,date,score\n2330,2020-01-01,0.7\n", encoding="utf-8")
        stale_binding_rejected = False
        try:
            _validate_audit_source_artifact(
                source_record,
                expected_path=source_path,
                label="synthetic PIT Scores",
                project_root=Path(artifact_dir),
            )
        except ValueError as exc:
            stale_binding_rejected = "SHA256" in str(exc)
    check(
        "point_in_time_audit_is_bound_to_exact_source_artifact_hashes",
        (True, True),
        (exact_binding_accepted, stale_binding_rejected),
    )

    gate = derive_point_in_time_model_validation_gate({
        "metrics": {"pass_only_target": {
            "global_spearman": 0.3074, "mean_daily_spearman": 0.2370,
        }},
        "direction_summary": {
            "valid_year_count": 7,
            "positive_spearman_year_count": 7,
            "positive_spread_year_count": 7,
        },
    })
    check(
        "point_in_time_model_gate_uses_only_target_ordering_evidence",
        ("PASS", False, False),
        (gate["status"], gate["strategy_metrics_used"], gate["future_target_used_for_runtime_sort"]),
    )

    diagnostic_orderable_input = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-03",
         "signal_date": "2020-01-02", "breakout_quality_score_date": "2020-01-01",
         "breakout_quality_score": 0.8, "breakout_quality_score_available": True},
        {"ticker": "B", "trade_date": "2020-01-03",
         "signal_date": "2020-01-01", "breakout_quality_score_date": "2020-01-01",
         "breakout_quality_score": 0.2, "breakout_quality_score_available": True},
    ])
    diagnostic_selected_input = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-03",
         "signal_date": "2020-01-02", "type": "買進"},
    ])
    diagnostic_lookup_input = pd.DataFrame([
        {"ticker": "A", "signal_date": "2020-01-01",
         "breakout_quality_score": 0.8, "target_raw_r": 2.0,
         "target_available": True},
        {"ticker": "B", "signal_date": "2020-01-01",
         "breakout_quality_score": 0.2, "target_raw_r": 1.0,
         "target_available": True},
    ])
    diagnostic_metrics, diagnostic_orderable, diagnostic_selected = (
        _strategy_selection_diagnostics(
            orderable=diagnostic_orderable_input,
            selected=diagnostic_selected_input,
            lookup=diagnostic_lookup_input,
        )
    )
    check(
        "point_in_time_strategy_diagnostic_uses_original_score_date_after_replay",
        (1.0, 1.0, 0.0, False, "2020-01-01", "2020-01-01"),
        (
                    diagnostic_metrics["orderable_score_coverage_rate"],
                    diagnostic_metrics["target_top_k_retention_mean"],
                    diagnostic_metrics["target_opportunity_gap_r_mean"],
                    diagnostic_metrics["future_target_used_for_runtime_sort"],
                    diagnostic_orderable.loc[0, "score_event_date"],
                    diagnostic_selected.loc[0, "score_event_date"],
                ),
    )

    true_score_mismatch_rejected = False
    mismatch_message_has_identity = False
    mismatched_orderable = diagnostic_orderable_input.copy()
    mismatched_orderable.loc[0, "breakout_quality_score"] = 0.7
    try:
        _strategy_selection_diagnostics(
            orderable=mismatched_orderable,
            selected=diagnostic_selected_input,
            lookup=diagnostic_lookup_input,
        )
    except ValueError as exc:
        true_score_mismatch_rejected = True
        mismatch_message_has_identity = all(
            text in str(exc)
            for text in (
                "ticker=A",
                "signal_date=2020-01-02",
                "score_event_date=2020-01-01",
                "runtime_score=0.7",
                "pit_score=0.8",
            )
        )
    check(
        "point_in_time_strategy_diagnostic_rejects_true_score_mismatch_with_identity",
        (True, True),
        (true_score_mismatch_rejected, mismatch_message_has_identity),
    )

    summary["workflow"] = "selection_point_in_time_scores"
    summary["score_contract"] = "future_target_excluded"
    summary["strategy_score_sort"] = "missing_fallback_original_buy_sort"
    return results, summary


def validate_breakout_quality_pit_training_performance_semantics_case(_base_params):
    """Protect PIT feeding/selection optimizations as execution-only semantics."""

    case_id = "BREAKOUT_QUALITY_PIT_TRAINING_PERFORMANCE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from config.breakout_quality import (
        BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS,
        DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
        get_breakout_quality_workflow_settings,
    )
    from filters.breakout_quality.features import (
        normalize_ohlcv_array_window,
        normalize_ohlcv_array_windows,
    )
    from filters.breakout_quality.models.runtime import require_torch
    from filters.breakout_quality.torch_runtime import resolve_torch_execution_plan
    from services.breakout_quality.train_continuous_ranker import (
        epoch_selection_metrics,
        parse_args as parse_ranker_args,
        select_epoch as select_ranker_epoch,
        split_metrics,
    )

    windows = np.asarray(
        [
            [
                [10.0, 11.0, 9.0, 10.0, 100.0],
                [11.0, 12.0, 10.0, 11.0, 200.0],
                [12.0, 13.0, 11.0, 12.0, 300.0],
                [13.0, 14.0, 12.0, 13.0, 400.0],
            ],
            [
                [20.0, 21.0, 19.0, 20.0, 50.0],
                [19.0, 20.0, 18.0, 19.0, 50.0],
                [18.0, 19.0, 17.0, 18.0, 50.0],
                [17.0, 18.0, 16.0, 17.0, 50.0],
            ],
        ],
        dtype=np.float64,
    )
    anchors = np.asarray([13.0, 17.0], dtype=np.float64)
    batch_features = normalize_ohlcv_array_windows(windows, anchors)
    scalar_features = np.stack(
        [
            normalize_ohlcv_array_window(windows[index], float(anchors[index]))
            for index in range(len(windows))
        ]
    )
    check_true(
        "pit_vectorized_ohlcv_normalization_is_byte_exact_to_scalar_contract",
        np.array_equal(batch_features, scalar_features),
    )

    days = 6
    per_day = 12
    count = days * per_day
    rng = np.random.default_rng(20260828)
    dates = np.repeat(
        pd.date_range("2025-01-01", periods=days, freq="D").to_numpy(),
        per_day,
    )
    raw = rng.normal(size=count).astype(np.float32)
    scores = (0.20 * raw + rng.normal(size=count)).astype(np.float32)
    percentile = np.empty(count, dtype=np.float32)
    for day_index in range(days):
        start = day_index * per_day
        stop = start + per_day
        percentile[start:stop] = (
            pd.Series(raw[start:stop])
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float32)
        )
    group_ids = np.arange(count, dtype=np.int64)
    group_table = pd.DataFrame(
        {
            "date": dates,
            "label": rng.integers(0, 2, size=count),
        }
    )
    full = split_metrics(group_ids, group_table, raw, percentile, scores)
    light = epoch_selection_metrics(
        group_ids, group_table, raw, percentile, scores
    )
    check(
        "pit_lightweight_epoch_metrics_preserve_exact_selection_values",
        (
            full["mean_daily_spearman"],
            full["median_daily_spearman"],
            full["rankable_date_count"],
            full["mse_vs_daily_percentile"],
        ),
        (
            light["mean_daily_spearman"],
            light["median_daily_spearman"],
            light["rankable_date_count"],
            light["mse_vs_daily_percentile"],
        ),
    )

    profile_name = DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    profile = get_breakout_quality_experiment_profile(profile_name)
    resolved_workflow = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
    args = parse_ranker_args(
        [
            "--experiment-profile",
            profile_name,
            "--model-architecture",
            resolved_workflow.model_architecture,
            "--epochs",
            "1",
            "--batch-size",
            "128",
            "--evaluation-batch-size",
            "64",
            "--device",
            "cpu",
            "--no-mixed-precision",
            "--train-prefetch-batches",
            "0",
        ]
    )
    torch, _nn = require_torch()
    torch.set_num_threads(1)
    plan = resolve_torch_execution_plan(
        torch,
        requested_device="cpu",
        mixed_precision=False,
        mixed_precision_dtype="auto",
        deterministic_algorithms=True,
        allow_tf32=False,
    )
    model_days = 4
    model_per_day = 8
    model_count = model_days * model_per_day
    model_rng = np.random.default_rng(20260829)
    features = model_rng.normal(size=(model_count, 300, 10)).astype(np.float32)
    context = np.empty((model_count, 0), dtype=np.float32)
    model_dates = np.repeat(
        pd.date_range("2025-02-01", periods=model_days, freq="D").to_numpy(),
        model_per_day,
    )
    favorable = model_rng.normal(2.0, 1.0, size=model_count).astype(np.float32)
    adverse = np.abs(model_rng.normal(0.5, 0.2, size=model_count)).astype(np.float32)
    model_table = pd.DataFrame(
        {
            "date": model_dates,
            "target_favorable_r": favorable,
            "target_adverse_r": adverse,
            "label": model_rng.integers(0, 2, size=model_count),
        }
    )
    model_pct = np.empty(model_count, dtype=np.float32)
    for day_index in range(model_days):
        start = day_index * model_per_day
        stop = start + model_per_day
        model_pct[start:stop] = (
            pd.Series(favorable[start:stop])
            .rank(method="average", pct=True)
            .to_numpy(dtype=np.float32)
        )
    train_ids = np.arange(0, 3 * model_per_day, dtype=np.int64)
    validation_ids = np.arange(3 * model_per_day, model_count, dtype=np.int64)
    full_selection = select_ranker_epoch(
        torch,
        features,
        context,
        model_table,
        favorable,
        model_pct,
        train_ids,
        validation_ids,
        args=args,
        plan=plan,
        evaluate_train_metrics=False,
        selection_metrics_only=False,
    )
    lightweight_selection = select_ranker_epoch(
        torch,
        features,
        context,
        model_table,
        favorable,
        model_pct,
        train_ids,
        validation_ids,
        args=args,
        plan=plan,
        evaluate_train_metrics=False,
        selection_metrics_only=True,
    )
    check(
        "pit_lightweight_selection_preserves_optimizer_loss_epoch_and_gate_metric",
        (
            full_selection["best_epoch"],
            full_selection["best_validation_mean_daily_spearman"],
            full_selection["best_validation_mse"],
            [row["batch_loss"] for row in full_selection["history"]],
        ),
        (
            lightweight_selection["best_epoch"],
            lightweight_selection["best_validation_mean_daily_spearman"],
            lightweight_selection["best_validation_mse"],
            [row["batch_loss"] for row in lightweight_selection["history"]],
        ),
    )
    check_true(
        "pit_lightweight_selection_is_config_driven_execution_only",
        isinstance(BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS, bool)
        and lightweight_selection.get("selection_metrics_only") is True,
    )

    project_root = Path(__file__).resolve().parents[2]
    pit_source = (
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    check_true(
        "pit_builder_routes_configured_lightweight_metrics_without_training_semantic_change",
        "BREAKOUT_QUALITY_PIT_EPOCH_SELECTION_LIGHTWEIGHT_METRICS" in pit_source
        and "selection_metrics_only=bool(" in pit_source,
    )

    summary["workflow"] = "pit_execution_performance"
    summary["scientific_change"] = False
    return results, summary
