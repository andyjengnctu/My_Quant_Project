from __future__ import annotations

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    COMPARISON_MODE_HARD_FILTER,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
    Path,
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
    SimpleNamespace,
    StrategyAlignedContinuousTargetSpec,
    TARGET_MANIFEST_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_TRADE_MATCHES_CSV_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    TRAINING_LABEL_SCOPE_ALL,
    TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
    TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
    _load_round_trip_source,
    _registered_breakout_quality_audit_module,
    _resolve_round_trip_path,
    add_check,
    ast,
    build_continuous_target_audit,
    build_daily_percentile_targets,
    build_file_manifest,
    build_strategy_aligned_group_targets,
    canonical_strategy_compare_output_dir_names,
    continuous_ranker_scope_group_ids,
    continuous_ranker_trade_alignment_metrics,
    get_breakout_quality_experiment_profile,
    json,
    load_validated_continuous_target_arrays,
    math,
    np,
    patch,
    pd,
    render_continuous_target_audit_markdown,
    resolve_continuous_target_dir,
    resolve_filter_artifact_paths,
    resolve_filter_model_output_dir,
    strategy_aligned_target_from_cached_path,
    tempfile,
)

from .source_index import read_source_ast, read_source_text

def validate_breakout_quality_continuous_target_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_TARGET"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    spec = StrategyAlignedContinuousTargetSpec.from_label_policy(DEFAULT_LABEL_POLICY)
    contract = spec.contract_payload()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_id_is_versioned",
        STRATEGY_ALIGNED_TARGET_ID,
        contract["target_id"],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_uses_no_split_or_oos_parameters",
        (False, False, "none", "none"),
        (
            contract["split_derived_parameters"],
            contract["oos_derived_parameters"],
            contract["normalization"],
            contract["clipping"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_inherits_current_horizon_and_risk_budget",
        (
            int(DEFAULT_LABEL_POLICY.label_horizon_bars),
            abs(float(DEFAULT_LABEL_POLICY.max_adverse_return)),
            float(DEFAULT_LABEL_POLICY.min_mfe_return)
            / abs(float(DEFAULT_LABEL_POLICY.max_adverse_return)),
        ),
        (
            int(contract["horizon_bars"]),
            float(contract["risk_budget_return"]),
            float(contract["full_horizon_time_penalty_r"]),
        ),
        tol=1e-12,
    )

    horizon = int(spec.horizon_bars)
    early_high = np.full(horizon, 101.0, dtype=np.float64)
    early_low = np.full(horizon, 98.0, dtype=np.float64)
    early_high[4:] = 120.0
    early = strategy_aligned_target_from_cached_path(
        early_high,
        early_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    risk_budget = float(spec.risk_budget_return)
    expected_early_target = (
        0.20 / risk_budget
        - 0.02 / risk_budget
        - float(spec.full_horizon_time_penalty_r) * (4.0 / float(horizon - 1))
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_rewards_fast_favorable_move_net_of_adverse_path",
        (True, 5, 0.20, 0.02, round(expected_early_target, 12)),
        (
            bool(early.valid),
            int(early.opportunity_bar),
            round(float(early.favorable_return), 12),
            round(float(early.adverse_return_to_peak), 12),
            round(float(early.target_raw_r), 12),
        ),
        tol=1e-9,
    )

    late_high = np.linspace(100.0, 105.0, horizon, dtype=np.float64)
    late_low = np.full(horizon, 100.0, dtype=np.float64)
    late = strategy_aligned_target_from_cached_path(
        late_high,
        late_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_time_penalty_offsets_minimum_move_at_horizon_end",
        (True, horizon, 0.0),
        (bool(late.valid), int(late.opportunity_bar), round(float(late.target_raw_r), 12)),
        tol=1e-9,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_prefers_early_larger_opportunity",
        True,
        float(early.target_raw_r) > float(late.target_raw_r),
    )

    barrier_high = np.full(horizon, 130.0, dtype=np.float64)
    barrier_low = np.full(horizon, 100.0, dtype=np.float64)
    barrier_low[0] = 90.0
    adverse_first = strategy_aligned_target_from_cached_path(
        barrier_high,
        barrier_low,
        anchor_price=100.0,
        available_bars=horizon,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_same_bar_risk_touch_excludes_high",
        (True, 0.0, 0.10, 1, 1, -1.0),
        (
            bool(adverse_first.valid),
            float(adverse_first.favorable_return),
            float(adverse_first.adverse_return_to_peak),
            int(adverse_first.opportunity_bar),
            int(adverse_first.first_risk_breach_bar),
            float(adverse_first.target_raw_r),
        ),
        tol=1e-12,
    )

    insufficient = strategy_aligned_target_from_cached_path(
        early_high,
        early_low,
        anchor_price=100.0,
        available_bars=horizon - 1,
        spec=spec,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_insufficient_future_is_invalid",
        (False, "insufficient_future", True),
        (
            bool(insufficient.valid),
            str(insufficient.reason),
            bool(math.isnan(insufficient.target_raw_r)),
        ),
    )

    anchors = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float64)
    future_highs = np.stack([early_high, late_high, barrier_high, early_high])
    future_lows = np.stack([early_low, late_low, barrier_low, early_low])
    available = np.array([horizon, horizon, horizon, horizon - 1], dtype=np.int64)
    target_arrays = build_strategy_aligned_group_targets(
        anchors,
        future_highs,
        future_lows,
        available,
        spec=spec,
    )
    repeated_arrays = build_strategy_aligned_group_targets(
        anchors,
        future_highs,
        future_lows,
        available,
        spec=spec,
    )
    deterministic = all(
        np.array_equal(target_arrays[key], repeated_arrays[key], equal_nan=True)
        for key in target_arrays
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_group_arrays_are_deterministic_and_group_scoped",
        ((4,), np.dtype(np.float32), np.dtype(np.bool_), [True, True, True, False], True),
        (
            target_arrays["target_raw_r"].shape,
            target_arrays["target_raw_r"].dtype,
            target_arrays["valid_mask"].dtype,
            target_arrays["valid_mask"].tolist(),
            deterministic,
        ),
    )

    dates = pd.to_datetime(
        [
            "2018-01-02", "2018-01-02", "2019-01-02", "2019-01-02",
            "2020-01-02", "2020-01-02", "2021-01-04", "2021-01-04",
        ]
    )
    synthetic_groups = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "date": dates,
            "label": [LABEL_PASS, LABEL_REJECT] * 4,
            "target_raw_r": [1.5, -0.8, 1.2, -0.6, 1.0, -0.4, 0.8, -0.2],
            "valid_mask": [True] * 8,
            "max_upside_return": [0.20, 0.01, 0.16, 0.02, 0.13, 0.03, 0.10, 0.04],
            "decision_mfe_return": [0.20, 0.01, 0.16, 0.02, 0.13, 0.03, 0.10, 0.04],
            "decision_mae_return": [-0.02, -0.08, -0.02, -0.07, -0.03, -0.06, -0.03, -0.05],
            "is_inner_train": [True, True, True, True, False, False, False, False],
            "is_validation": [False, False, False, False, True, True, False, False],
            "is_selection": [True, True, True, True, True, True, False, False],
            "is_oos": [False, False, False, False, False, False, True, True],
        }
    )
    audit_payload, daily = build_continuous_target_audit(
        synthetic_groups,
        target_contract=contract,
        split_report={"policy": "synthetic_fixed_split"},
        dataset_summary={
            "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            "dataset": "synthetic",
            "event_count": len(synthetic_groups),
            "feature_group_count": len(synthetic_groups),
        },
        trade_alignment={
            "available": False,
            "reason": "synthetic_no_round_trip_file",
            "formula_tuned_from_trade_r": False,
            "diagnostic_only": True,
        },
    )
    markdown = render_continuous_target_audit_markdown(audit_payload)
    strict_json_ok = True
    try:
        json.dumps(audit_payload, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        strict_json_ok = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audit_is_rankable_strict_json_and_audit_only",
        ("IMPLEMENTED_AUDIT_ONLY", False, 1.0, 1.0, 7, True, True),
        (
            audit_payload["status"],
            audit_payload["training_performed"],
            audit_payload["split_metrics"]["oos"]["same_day_rankability"]["rankable_date_rate"],
            audit_payload["split_metrics"]["selection"]["same_day_binary_concordance"]["pair_weighted_concordance"],
            len(daily),
            strict_json_ok,
            "本輪沒有訓練模型" in markdown,
        ),
        tol=1e-12,
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        preferred_hard_filter_dir = canonical_strategy_compare_output_dir_names(
            COMPARISON_MODE_HARD_FILTER
        )[0]
        expected_round_trip_path = (
            resolve_filter_model_output_dir(
                temp_dir,
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
            )
            / preferred_hard_filter_dir
            / "no_filter_round_trips.csv"
        )
        expected_round_trip_path.parent.mkdir(parents=True, exist_ok=True)
        expected_round_trip_path.write_text(
            "ticker,r_multiple\n2330,1.0\n",
            encoding="utf-8",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            resolved_round_trip_path, resolved_path_source = _resolve_round_trip_path(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_round_trip_auto_path_uses_output_tree",
            (str(expected_round_trip_path), "active_9a_standard_path"),
            (str(resolved_round_trip_path), str(resolved_path_source)),
        )

        expected_round_trip_path.unlink()
        trade_history_path = expected_round_trip_path.parent / "no_filter_trades.csv"
        trade_history_frame = pd.DataFrame(
            [
                {
                    "Date": "2021-01-05",
                    "Ticker": "2330",
                    "Type": "買進 (突破)",
                    "進場類型": "normal",
                    "候選類型": "normal",
                    "買訊日": "2021-01-04",
                    "候選日": "2021-01-04",
                    "成交價": 100.0,
                    "該筆總損益": np.nan,
                    "R_Multiple": np.nan,
                },
                {
                    "Date": "2021-01-15",
                    "Ticker": "2330",
                    "Type": "全倉結算",
                    "進場類型": "",
                    "候選類型": "",
                    "買訊日": "",
                    "候選日": "",
                    "成交價": 110.0,
                    "該筆總損益": 1000.0,
                    "R_Multiple": 1.5,
                },
            ]
        )
        trade_history_frame.to_csv(trade_history_path, index=False, encoding="utf-8-sig")
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            rebuilt_round_trips, rebuilt_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        rebuilt_record = rebuilt_round_trips.iloc[0]
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_round_trip_falls_back_to_canonical_trade_history",
            (
                1,
                "2330",
                "2021-01-04",
                1.5,
                "reconstructed_from_no_filter_trades",
                True,
                str(trade_history_path),
            ),
            (
                len(rebuilt_round_trips),
                str(rebuilt_record["ticker"]),
                str(rebuilt_record["signal_date"]),
                float(rebuilt_record["r_multiple"]),
                str(rebuilt_source["source_kind"]),
                bool(rebuilt_source["round_trips_reconstructed"]),
                str(rebuilt_source["path"]),
            ),
            tol=1e-12,
        )


        trade_history_path.unlink()
        active_output_root = expected_round_trip_path.parents[1]
        skipped_static_dir = active_output_root / "strategy_compare_score_ranking_base_finalists_agree"
        skipped_static_dir.mkdir(parents=True, exist_ok=True)
        (skipped_static_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                        "comparison_design": "static_param_diagnostic",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        trade_history_frame.to_csv(
            skipped_static_dir / "no_filter_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        discovered_dir = active_output_root / "strategy_compare_score_ranking_base_finalist_best"
        discovered_dir.mkdir(parents=True, exist_ok=True)
        (discovered_dir / "strategy_comparison.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                        "comparison_design": "historical_active_param_oos",
                    }
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        discovered_trade_history_path = discovered_dir / "no_filter_trades.csv"
        trade_history_frame.to_csv(
            discovered_trade_history_path,
            index=False,
            encoding="utf-8-sig",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            discovered_round_trips, discovered_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_discovers_official_score_ranking_trade_history_and_skips_static",
            (
                1,
                "active_9a_strategy_compare_trade_history_discovery",
                str(discovered_trade_history_path),
                "historical_active_param_oos",
            ),
            (
                len(discovered_round_trips),
                str(discovered_source["path_source"]),
                str(discovered_source["path"]),
                str((discovered_source.get("strategy_compare_metadata") or {}).get("comparison_design")),
            ),
        )

        explicit_trade_history_path = Path(temp_dir) / "explicit_no_filter_trades.csv"
        trade_history_frame.to_csv(
            explicit_trade_history_path,
            index=False,
            encoding="utf-8-sig",
        )
        with patch(
            "tools.audit.breakout_quality.continuous_target.PROJECT_ROOT",
            Path(temp_dir),
        ):
            explicit_round_trips, explicit_source = _load_round_trip_source(
                BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                None,
                str(explicit_trade_history_path),
            )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_target_accepts_explicit_trade_history_override",
            (1, "explicit_trade_history", str(explicit_trade_history_path), True),
            (
                len(explicit_round_trips),
                str(explicit_source["path_source"]),
                str(explicit_source["path"]),
                bool(explicit_source["round_trips_reconstructed"]),
            ),
        )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_output_path_is_target_version_scoped",
        (
            "breakout_quality_v1",
            "continuous_targets",
            STRATEGY_ALIGNED_TARGET_ID,
        ),
        tuple(
            resolve_continuous_target_dir(
                Path("/tmp/project"),
                "breakout_quality_v1",
            ).parts[-3:]
        ),
    )

    breakout_quality_app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    breakout_quality_app_tree = ast.parse(
        breakout_quality_app_path.read_text(encoding="utf-8"),
        filename=str(breakout_quality_app_path),
    )
    command_modules = {}
    for node in breakout_quality_app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES"
            for target in node.targets
        ):
            continue
        command_modules = ast.literal_eval(node.value)
        break

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audit_command_is_registered",
        "tools.audit.breakout_quality.continuous_target",
        _registered_breakout_quality_audit_module("audit-continuous-target"),
    )

    summary["target_id"] = STRATEGY_ALIGNED_TARGET_ID
    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_continuous_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CONTINUOUS_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_named_training_profile_not_model_architecture",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_TARGET_ID,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    pass_profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_is_named_profile_with_no_time_target_and_pass_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_PASS_ONLY,
            False,
        ),
        (
            pass_profile.training_objective,
            pass_profile.continuous_target_id,
            pass_profile.training_label_scope,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    raw = np.asarray([1.0, 3.0, 2.0, 5.0, 5.0, 9.0], dtype=np.float32)
    valid = np.ones((6,), dtype=bool)
    dates = pd.Series(["2020-01-02"] * 3 + ["2021-05-03"] * 3)
    percentiles = build_daily_percentile_targets(raw, valid, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_daily_percentile_spans_zero_one_and_averages_ties",
        (0.0, 1.0, 0.5, 0.25, 0.25, 1.0),
        tuple(round(float(value), 6) for value in percentiles),
    )

    synthetic_group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_PASS],
            "date": pd.to_datetime(dates),
        }
    )
    scoped_ids = continuous_ranker_scope_group_ids(
        np.arange(6, dtype=np.int64),
        synthetic_group_table,
        label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    )
    pass_mask = np.zeros((6,), dtype=bool)
    pass_mask[scoped_ids] = True
    pass_percentiles = build_daily_percentile_targets(raw, pass_mask, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_percentiles_use_only_same_date_pass_groups",
        (0, 2, 4, 5, 0.0, 1.0, 0.0, 1.0, True, True),
        (
            int(scoped_ids[0]), int(scoped_ids[1]), int(scoped_ids[2]), int(scoped_ids[3]),
            round(float(pass_percentiles[0]), 6),
            round(float(pass_percentiles[2]), 6),
            round(float(pass_percentiles[4]), 6),
            round(float(pass_percentiles[5]), 6),
            bool(np.isnan(pass_percentiles[1])),
            bool(np.isnan(pass_percentiles[3])),
        ),
    )

    singleton_percentile = build_daily_percentile_targets(
        np.asarray([7.0], dtype=np.float32),
        np.asarray([True], dtype=bool),
        pd.Series(["2022-08-08"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_singleton_date_uses_neutral_half_percentile",
        (0.5,),
        tuple(float(value) for value in singleton_percentile),
    )

    changed_oos = raw.copy()
    changed_oos[3:] = np.asarray([-100.0, 500.0, 0.0], dtype=np.float32)
    changed_percentiles = build_daily_percentile_targets(changed_oos, valid, dates)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_same_date_transform_prevents_cross_split_distribution_leakage",
        tuple(percentiles[:3]),
        tuple(changed_percentiles[:3]),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        target_dir = resolve_continuous_target_dir(
            root,
            "synthetic_quality",
            target_id=STRATEGY_ALIGNED_TARGET_ID,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_path = target_dir / TARGET_RAW_FILENAME
        valid_path = target_dir / TARGET_VALID_MASK_FILENAME
        np.save(raw_path, raw, allow_pickle=False)
        np.save(valid_path, valid, allow_pickle=False)
        dataset_source_path = root / "events.csv"
        dataset_source_path.write_text("ticker,date\n2330,2020-01-02\n", encoding="utf-8")
        dataset_artifacts = {"events_csv": build_file_manifest(dataset_source_path)}
        manifest = {
            "schema_version": 1,
            "filter_id": "synthetic_quality",
            "target_contract": {"target_id": STRATEGY_ALIGNED_TARGET_ID},
            "group_count": 6,
            "dataset_policy": {"policy": "synthetic"},
            "dataset_artifact_source": dataset_artifacts,
            "artifacts": {
                "target_raw_r": build_file_manifest(raw_path),
                "valid_mask": build_file_manifest(valid_path),
            },
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        loaded_manifest, loaded_raw, loaded_valid = load_validated_continuous_target_arrays(
            root,
            "synthetic_quality",
            expected_group_count=6,
            expected_dataset_policy={"policy": "synthetic"},
            expected_dataset_artifacts=dataset_artifacts,
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_strictly_loads_versioned_target_arrays",
            (STRATEGY_ALIGNED_TARGET_ID, tuple(raw), tuple(valid)),
            (
                loaded_manifest["target_contract"]["target_id"],
                tuple(loaded_raw),
                tuple(loaded_valid),
            ),
        )
        stale_dataset_artifacts = {
            "events_csv": {
                **dataset_artifacts["events_csv"],
                "sha256": "0" * 64,
            }
        }
        try:
            load_validated_continuous_target_arrays(
                root,
                "synthetic_quality",
                expected_group_count=6,
                expected_dataset_policy={"policy": "synthetic"},
                expected_dataset_artifacts=stale_dataset_artifacts,
            )
            stale_dataset_rejected = False
        except ValueError as exc:
            stale_dataset_rejected = "Dataset artifact" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_rejects_target_from_different_dataset_artifacts",
            True,
            stale_dataset_rejected,
        )

        raw_path.write_bytes(raw_path.read_bytes() + b"tamper")
        try:
            load_validated_continuous_target_arrays(
                root,
                "synthetic_quality",
                expected_group_count=6,
                expected_dataset_policy={"policy": "synthetic"},
            )
            tamper_rejected = False
        except ValueError as exc:
            tamper_rejected = "size" in str(exc) or "SHA256" in str(exc)
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "continuous_ranker_rejects_tampered_target_artifact",
            True,
            tamper_rejected,
        )

    app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    tree = read_source_ast(app_path)
    command_modules = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_research_command_is_registered",
        "services.breakout_quality.ranker_cli",
        command_modules.get("train-continuous-ranker"),
    )

    project_root = Path(__file__).resolve().parents[2]
    ranker_source = (
        project_root
        / "services"
        / "breakout_quality"
        / "train_continuous_ranker.py"
    ).read_text(encoding="utf-8")
    inception_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "models"
        / "inception_time.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_reuses_active_two_logit_head_and_pass_probability",
        (True, True, True),
        (
            "self.classifier = nn.Linear(module_output_channels, 2)" in inception_source,
            "torch.softmax(logits.float(), dim=1)[:, LABEL_PASS]" in ranker_source,
            '"model_state_dict"' in ranker_source and "torch.save(" in ranker_source,
        ),
    )
    train_source = (
        project_root
        / "services"
        / "breakout_quality"
        / "train.py"
    ).read_text(encoding="utf-8")
    artifact_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "artifacts.py"
    ).read_text(encoding="utf-8")
    app_source = app_path.read_text(encoding="utf-8")
    export_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "export_scores.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_cli_only_and_does_not_pollute_interactive_menu",
        (True, True, True, True),
        (
            command_modules.get("train-continuous-ranker")
            == "services.breakout_quality.ranker_cli",
            'print("[10] 11B 同日 Percentile Ranker（research-only）")' not in app_source,
            'elif choice == "10":' not in app_source,
            "_interactive_train_continuous_ranker" not in app_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_profile_is_blocked_from_binary_workflow_and_runtime_loader",
        (True, True, True, True),
        (
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in train_source,
            "research-only continuous ranker artifact不得載入正式binary runtime contract" in artifact_source,
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in app_source,
            "SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES" in export_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_is_research_only_and_oos_follows_checkpoint_write",
        (True, True, True, True, True, True),
        (
            '"eligible": False' in ranker_source,
            "OOS target transformation and model inference occur only after" in ranker_source,
            "torch.save(" in ranker_source
            and ranker_source.index("torch.save(")
            < ranker_source.index("OOS target transformation and model inference occur only after"),
            'score_frame["group_index"].duplicated().any()' in ranker_source,
            'label_scope=profile.training_label_scope' in ranker_source,
            '"label_conditional": {}' in ranker_source,
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_is_cli_only_and_uses_existing_command",
        (True, True, True, True),
        (
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in ranker_source,
            'choices=(' in ranker_source,
            "11G" not in app_source[app_source.index("def _interactive_model_research"):app_source.index("def run_model_training_menu")],
            command_modules.get("train-continuous-ranker")
            == "services.breakout_quality.ranker_cli",
        ),
    )

    ranker_api_source = (
        project_root / "services" / "breakout_quality" / "ranker_training.py"
    ).read_text(encoding="utf-8")
    pipeline_source = (
        project_root / "services" / "breakout_quality" / "continuous_ranker_pipeline.py"
    ).read_text(encoding="utf-8")
    daily_source = (
        project_root / "services" / "breakout_quality" / "train_daily_ranker.py"
    ).read_text(encoding="utf-8")
    public_api_names = (
        "select_epoch",
        "fit_final",
        "predict_scores",
        "split_metrics",
        "training_semantics",
        "resolve_training_output_paths",
        "daily_rank_metrics",
        "daily_top_k_metrics",
        "calculate_spearman",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ranker_training_public_api_is_single_shared_boundary_without_private_cross_module_calls",
        True,
        all(name in ranker_api_source for name in public_api_names)
        and "ranker_training as ranker_api" in pipeline_source
        and "ranker_training as ranker_api" in daily_source
        and "ranker_impl._" not in pipeline_source
        and "ranker_impl._" not in daily_source
        and "ranker_impl=" not in daily_source
        and "ranker_impl=sys.modules" not in ranker_source,
    )

    from services.breakout_quality import ranker_training as ranker_training_api
    from services.breakout_quality import train_continuous_ranker as canonical_ranker
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_run_does_not_rebind_shared_public_api_names",
        True,
        not (set(public_api_names) & set(canonical_ranker.run.__code__.co_varnames)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "legacy_private_ranker_names_alias_the_public_training_api_without_second_implementation",
        True,
        canonical_ranker._select_epoch is ranker_training_api.select_epoch
        and canonical_ranker._fit_final is ranker_training_api.fit_final
        and canonical_ranker._predict_scores is ranker_training_api.predict_scores
        and canonical_ranker._split_metrics is ranker_training_api.split_metrics
        and canonical_ranker._training_semantics is ranker_training_api.training_semantics
        and canonical_ranker._training_output_paths is ranker_training_api.resolve_training_output_paths
        and canonical_ranker._daily_rank_metrics is ranker_training_api.daily_rank_metrics
        and canonical_ranker._daily_top_k_metrics is ranker_training_api.daily_top_k_metrics
        and canonical_ranker._spearman is ranker_training_api.calculate_spearman,
    )

    summary["profile"] = STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE
    summary["pass_conditional_profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["training_objective"] = profile.training_objective
    return results, summary

def validate_breakout_quality_pass_conditional_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PASS_CONDITIONAL_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_profile_uses_no_time_target_and_pass_only_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_PASS_ONLY,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_PASS],
            "date": pd.to_datetime(["2020-01-02"] * 3 + ["2021-05-03"] * 3),
        }
    )
    ids = continuous_ranker_scope_group_ids(
        np.arange(6, dtype=np.int64),
        group_table,
        label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_scope_keeps_only_original_pass_groups",
        (0, 2, 4, 5),
        tuple(int(value) for value in ids),
    )

    raw = np.asarray([1.0, 99.0, 3.0, -50.0, 5.0, 9.0], dtype=np.float32)
    mask = np.zeros((6,), dtype=bool)
    mask[ids] = True
    percentiles = build_daily_percentile_targets(raw, mask, group_table["date"] )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_percentile_ignores_same_date_reject_target_values",
        (0.0, 1.0, 0.0, 1.0, True, True),
        (
            round(float(percentiles[0]), 6),
            round(float(percentiles[2]), 6),
            round(float(percentiles[4]), 6),
            round(float(percentiles[5]), 6),
            bool(np.isnan(percentiles[1])),
            bool(np.isnan(percentiles[3])),
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    tree = read_source_ast(app_path)
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES"
            for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")
    ]
    ranker_path = root / "services" / "breakout_quality" / "train_continuous_ranker.py"
    ranker_source = ranker_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_reuses_existing_cli_and_is_not_in_menu",
        (True, True, True, True),
        (
            command_modules.get("train-continuous-ranker")
            == "services.breakout_quality.ranker_cli",
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in ranker_source,
            "11G" not in menu_source,
            "strategy_aligned_no_time_pass_magnitude_mse" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_freezes_checkpoint_before_oos_pass_percentile",
        (True, True, True),
        (
            "torch.save(" in ranker_source,
            ranker_source.index("torch.save(")
            < ranker_source.index("oos_target_mask = np.zeros"),
            'label_scope=profile.training_label_scope' in ranker_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_outputs_label_conditional_trade_diagnostics",
        (True, True, True, True),
        (
            '"label_conditional": {}' in ranker_source,
            'for label_name, label_value in (("PASS", LABEL_PASS), ("REJECT", LABEL_REJECT))' in ranker_source,
            '"spearman_model_score_vs_target"' in ranker_source,
            '"spearman_target_vs_r_multiple"' in ranker_source,
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        target_dir = Path(temp_dir)
        pd.DataFrame(
            [
                {"ticker": "2330", "target_date": "2021-01-04", "r_multiple": 2.0, "target_raw_r": 99.0},
                {"ticker": "2317", "target_date": "2021-01-04", "r_multiple": -1.0, "target_raw_r": 99.0},
                {"ticker": "2454", "target_date": "2021-01-05", "r_multiple": 3.0, "target_raw_r": 99.0},
            ]
        ).to_csv(
            target_dir / TARGET_TRADE_MATCHES_CSV_FILENAME,
            index=False,
            encoding="utf-8-sig",
        )
        score_frame = pd.DataFrame(
            [
                {"ticker": "2330", "date": "2021-01-04", "label": LABEL_PASS, "target_raw_r": 2.5, "model_score": 0.8},
                {"ticker": "2317", "date": "2021-01-04", "label": LABEL_REJECT, "target_raw_r": -0.2, "model_score": 0.2},
                {"ticker": "2454", "date": "2021-01-05", "label": LABEL_PASS, "target_raw_r": 3.2, "model_score": 0.9},
            ]
        )
        trade_metrics = continuous_ranker_trade_alignment_metrics(score_frame, target_dir)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_trade_alignment_replaces_old_target_and_splits_labels",
        (3, 2, 1, 1.0),
        (
            trade_metrics.get("matched_trade_count"),
            trade_metrics.get("label_conditional", {}).get("PASS", {}).get("matched_trade_count"),
            trade_metrics.get("label_conditional", {}).get("REJECT", {}).get("matched_trade_count"),
            round(float(trade_metrics.get("spearman_target_vs_r_multiple")), 6),
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_conditional_ranker_remains_research_only_without_runtime_combination",
        (True, True, True, True),
        (
            '"eligible": False' in ranker_source,
            '"scope": "research_only"' in ranker_source,
            "forward_oos" not in ranker_source[ranker_source.find('"runtime_eligibility"'):],
            "score_blend" not in ranker_source.lower(),
        ),
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["training_label_scope"] = TRAINING_LABEL_SCOPE_PASS_ONLY
    return results, summary

def validate_breakout_quality_all_event_no_time_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_ALL_EVENT_NO_TIME_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.filters.breakout_quality.train_continuous_ranker import (
        _profile_contract,
        _scope_group_ids,
        build_daily_percentile_targets,
        parse_args as parse_continuous_ranker_args,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_no_time_profile_uses_existing_target_and_all_label_scope",
        (
            TRAINING_OBJECTIVE_DAILY_PERCENTILE_REGRESSION,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
            TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
            "mse",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.training_sample_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    group_table = pd.DataFrame(
        {
            "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS],
            "date": pd.to_datetime(["2020-01-02"] * 3),
        }
    )
    ids = _scope_group_ids(
        np.arange(3, dtype=np.int64),
        group_table,
        label_scope=TRAINING_LABEL_SCOPE_ALL,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_scope_keeps_pass_and_reject_groups",
        (0, 1, 2),
        tuple(int(value) for value in ids),
    )

    raw = np.asarray([1.0, 9.0, 5.0], dtype=np.float32)
    pct = build_daily_percentile_targets(
        raw, np.ones(3, dtype=bool), group_table["date"]
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_event_daily_percentile_ranks_across_binary_labels",
        (0.0, 1.0, 0.5),
        tuple(float(value) for value in pct),
    )

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE]
    )
    contract = _profile_contract(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12a_cli_profile_identity_is_explicit_and_research_only",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE, "12A", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    summary["training_label_scope"] = TRAINING_LABEL_SCOPE_ALL
    return results, summary

def validate_breakout_quality_pairwise_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PAIRWISE_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from filters.breakout_quality.models.factory import require_torch
    from config.strategy_compare import get_strategy_comparison_settings
    from tools.filters.breakout_quality.train_continuous_ranker import (
        PAIRWISE_TRAINING_CONTRACT,
        _daily_top_k_metrics,
        _date_coherent_batches,
        _pairwise_logistic_loss,
        _profile_contract,
        _training_semantics,
        _trade_alignment_metrics,
        parse_args as parse_continuous_ranker_args,
    )
    import config.breakout_quality as breakout_quality_config
    from config.breakout_quality import (
        get_breakout_quality_continuous_ranker_comparison_settings,
    )
    from filters.breakout_quality.continuous_ranker_quality import (
        exact_random_top_k_baseline,
    )
    from tools.filters.breakout_quality.compare_continuous_rankers import (
        _load_model_frame,
        _dynamic_orderable_frame,
        _evaluate_dynamic_k_strata,
        _evaluate_fixed_k_sweep,
        _evaluate_paired_frame,
        _evaluate_reference_subset_attribution,
        _evaluate_score_event_date_comparability,
        _render_dynamic_k_strata_table,
        _render_reference_subset_attribution_table,
        _render_score_event_date_comparability_table,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12b_profile_changes_only_learning_objective_with_same_target_scope_architecture_family",
        (
            TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
            TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
            "pairwise_logistic",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.training_sample_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    semantics = _training_semantics(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12b_pairwise_artifact_semantics_are_explicit_and_runtime_score_is_unchanged",
        (
            "same_date_non_tied_target_pairs",
            "equal_pair_weight",
            "pass_logit_minus_reject_logit",
            "whole_date_pack_no_date_split",
            "softmax_pass_probability",
        ),
        tuple(
            semantics["pairwise_contract"][key]
            for key in ("pair_scope", "pair_weighting", "model_margin", "batching", "runtime_score")
        ),
    )
    assert semantics["pairwise_contract"] == PAIRWISE_TRAINING_CONTRACT

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE]
    )
    contract = _profile_contract(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12b_cli_and_registry_identity_are_explicit",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE, "12B", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    dates = pd.Series(["2024-01-02"] * 3 + ["2024-01-03"] * 2 + ["2024-01-04"] * 3)
    batches = _date_coherent_batches(
        np.arange(8, dtype=np.int64), dates, batch_size=4, seed=42
    )
    batch_by_group = {
        int(group_id): int(batch_index)
        for batch_index, batch in enumerate(batches)
        for group_id in batch
    }
    date_batch_counts = []
    for _date, day in pd.DataFrame({"date": dates, "group_id": np.arange(8)}).groupby("date"):
        date_batch_counts.append(len({batch_by_group[int(group_id)] for group_id in day["group_id"]}))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pairwise_training_batches_never_split_same_date_competition_set",
        (1, 1, 1),
        tuple(date_batch_counts),
    )

    torch, _nn = require_torch()
    targets = torch.tensor([0.0, 0.5, 1.0, 0.0, 1.0], dtype=torch.float32)
    loss_dates = np.asarray(["2024-01-02"] * 3 + ["2024-01-03"] * 2)
    good_margin = torch.tensor([-2.0, 0.0, 2.0, -1.0, 1.0], dtype=torch.float32, requires_grad=True)
    bad_margin = torch.tensor([2.0, 0.0, -2.0, 1.0, -1.0], dtype=torch.float32, requires_grad=True)
    good_loss, good_pairs = _pairwise_logistic_loss(torch, good_margin, targets, loss_dates)
    bad_loss, bad_pairs = _pairwise_logistic_loss(torch, bad_margin, targets, loss_dates)
    good_loss.backward()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pairwise_logistic_prefers_correct_within_day_order_and_has_finite_gradient",
        (True, 4, 4, True),
        (
            bool(float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())),
            int(good_pairs),
            int(bad_pairs),
            bool(torch.isfinite(good_margin.grad).all().item()),
        ),
    )

    report_dates = np.asarray(["2024-02-01"] * 6 + ["2024-02-02"] * 6 + ["2024-02-05"] * 2)
    report_raw_target = np.asarray([6, 5, 4, 3, 2, 1, 12, 10, 8, 6, 4, 2, 100, -100], dtype=np.float64)
    report_percentile = np.concatenate([
        np.tile(np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64), 2),
        np.asarray([0.0, 1.0], dtype=np.float64),
    ])
    report_metrics = _daily_top_k_metrics(
        report_dates,
        report_raw_target / float(report_raw_target.max()),
        report_raw_target,
        report_percentile,
        top_k=3,
        boundary_width=2,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_top_k_report_measures_perfect_order_and_k_boundary",
        (1.0, 1.0, 1.0, 2, 2, 1, True),
        (
            round(float(report_metrics["ndcg_at_k"]), 6),
            round(float(report_metrics["oracle_top_k_overlap"]), 6),
            round(float(report_metrics["boundary_concordance"]), 6),
            int(report_metrics["competition_date_count"]),
            int(report_metrics["boundary_date_count"]),
            int(report_metrics["excluded_non_competition_date_count"]),
            bool(float(report_metrics["boundary_raw_target_gap"]) > 0.0),
        ),
    )

    random_baseline = exact_random_top_k_baseline(
        np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64),
        np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
        top_k=3,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_random_baseline_is_exact_not_monte_carlo",
        (True, 0.0, 0.5, 0.5, 0.0),
        (
            bool(0.0 < float(random_baseline["ndcg_at_k"]) < 1.0),
            float(random_baseline["top_k_raw_target_lift"]),
            float(random_baseline["oracle_top_k_overlap"]),
            float(random_baseline["boundary_concordance"]),
            float(random_baseline["boundary_raw_target_gap"]),
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        score_path = root / "continuous_ranker_scores.csv"
        pd.DataFrame([
            {
                "ticker": "A", "date": "2021-01-04", "group_index": 0,
                "split": "oos", "target_raw_r": 0.2,
                "target_daily_percentile": 1.0, "model_score": 0.8,
            },
            {
                "ticker": "B", "date": "2021-01-04", "group_index": 1,
                "split": "oos", "target_raw_r": 0.1,
                "target_daily_percentile": 0.5, "model_score": 0.6,
            },
            {
                "ticker": "C", "date": "2021-12-31", "group_index": 2,
                "split": "oos", "target_raw_r": np.nan,
                "target_daily_percentile": np.nan, "model_score": 0.7,
            },
        ]).to_csv(score_path, index=False, encoding="utf-8-sig")
        fake_contract = SimpleNamespace(
            score_path=score_path,
            continuous_target_id=profile.continuous_target_id,
            report_path=root / "report.json",
            model_information_cutoff="2020-12-31",
            available_from="2021-01-04",
            available_through="2021-12-31",
        )
        with patch(
            "tools.filters.breakout_quality.compare_continuous_rankers.load_continuous_ranker_oos_contract",
            return_value=fake_contract,
        ):
            comparable_frame, _comparable_meta = _load_model_frame(
                root=root,
                filter_id="synthetic",
                architecture="synthetic",
                model_id="synthetic",
                profile=STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE,
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_quality_comparison_uses_target_evaluable_subset_not_runtime_score_universe",
        (["A", "B"], True),
        (
            comparable_frame["ticker"].tolist(),
            bool(np.isfinite(comparable_frame["target_raw_r"].to_numpy(dtype=np.float64)).all()),
        ),
    )

    with (
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_PROFILES",
            (
                ("CFG-A", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE),
                ("CFG-B", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE),
                ("CFG-C", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE),
            ),
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_REFERENCE_ARM",
            "CFG-REFERENCE",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_SUMMARY_PAIR",
            ("CFG-B", "CFG-A"),
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_MENU_LABEL",
            "Config comparison",
        ),
        patch.object(
            breakout_quality_config,
            "BREAKOUT_QUALITY_CONTINUOUS_RANKER_COMPARISON_FIXED_K_VALUES",
            (1, 4, 9),
        ),
    ):
        configured_comparison = (
            get_breakout_quality_continuous_ranker_comparison_settings()
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_p2_comparison_settings_follow_isolated_config_override",
        (
            ("CFG-A", "CFG-B", "CFG-C"),
            "CFG-REFERENCE",
            ("CFG-B", "CFG-A"),
            "Config comparison",
            (1, 4, 9),
        ),
        (
            configured_comparison.model_ids,
            configured_comparison.reference_arm,
            configured_comparison.summary_pair,
            configured_comparison.menu_label,
            configured_comparison.fixed_k_values,
        ),
    )

    paired_frame = pd.DataFrame(
        {
            "date": ["2024-03-01"] * 6,
            "ticker": [f"T{i}" for i in range(6)],
            "group_index": np.arange(6, dtype=np.int64),
            "split": ["oos"] * 6,
            "target_raw_r": np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
            "target_daily_percentile": np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.0], dtype=np.float64),
            "score__MR-12A": np.asarray([6, 4, 5, 3, 2, 1], dtype=np.float64),
            "score__MR-12B": np.asarray([6, 5, 4, 3, 2, 1], dtype=np.float64),
            "score__MR-12C": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64),
        }
    )
    paired_eval = _evaluate_paired_frame(
        paired_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        k_by_date={"2024-03-01": 3},
        boundary_width=2,
    )
    ba = paired_eval["paired_contrasts"]["MR-12B_minus_MR-12A"]["metrics"]
    cb = paired_eval["paired_contrasts"]["MR-12C_minus_MR-12B"]["metrics"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_p2_uses_same_day_paired_deltas",
        (1, True, True, 1.0),
        (
            int(paired_eval["competition_date_count"]),
            bool(float(ba["ndcg_at_k"]["mean_delta"]) > 0.0),
            bool(float(cb["ndcg_at_k"]["mean_delta"]) < 0.0),
            float(paired_eval["models"]["MR-12B"]["boundary_concordance"]),
        ),
    )

    fixed_prefix = _evaluate_fixed_k_sweep(
        paired_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        k_values=(1, 2, 3),
        boundary_width=2,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_p2_fixed_k_prefix_sweep_reuses_same_oos_candidate_universe",
        (("1", "2", "3"), 1, True, True),
        (
            tuple(sorted(fixed_prefix, key=int)),
            int(fixed_prefix["1"]["competition_date_count"]),
            bool(
                float(
                    fixed_prefix["1"]["paired_contrasts"]["MR-12B_minus_MR-12A"]["metrics"]["ndcg_at_k"]["mean_delta"]
                ) >= 0.0
            ),
            bool(
                float(
                    fixed_prefix["3"]["paired_contrasts"]["MR-12C_minus_MR-12B"]["metrics"]["ndcg_at_k"]["mean_delta"]
                ) < 0.0
            ),
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        pair_dir = Path(td)
        pd.DataFrame(
            [
                {
                    "Date": "2024-03-04",
                    "Resource_Aware_Max_DL_Eligible": True,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
                {
                    "Date": "2024-03-05",
                    "Resource_Aware_Max_DL_Eligible": True,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
                {
                    "Date": "2024-03-06",
                    "Resource_Aware_Max_DL_Eligible": False,
                    "Resource_Aware_Pre_Market_Order_Limit": 2,
                },
            ]
        ).to_csv(pair_dir / "score_ranking_daily_capacity.csv", index=False)
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "signal_date": signal_date,
                    "breakout_quality_score_date": "2024-03-01",
                    "breakout_quality_score": 0.75,
                    "breakout_quality_score_unavailable_reason": "",
                }
                for trade_date, signal_date in (
                    ("2024-03-04", "2024-03-01"),
                    ("2024-03-05", "2024-03-02"),
                )
                for ticker in ("A", "B", "C", "D")
            ]
        ).to_csv(pair_dir / "score_ranking_orderable_candidates.csv", index=False)
        dynamic_models = {}
        target_raw = [4.0, 3.0, 2.0, 1.0]
        target_pct = [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0]
        for model_id, scores in {
            "MR-12A": [0.8, 0.7, 0.6, 0.5],
            "MR-12B": [0.9, 0.8, 0.7, 0.6],
            "MR-12C": [0.6, 0.7, 0.8, 0.9],
        }.items():
            dynamic_models[model_id] = pd.DataFrame(
                {
                    "ticker": ["A", "B", "C", "D"],
                    "date": ["2024-03-01"] * 4,
                    "group_index": [1, 2, 3, 4],
                    "split": ["oos"] * 4,
                    "target_raw_r": target_raw,
                    "target_daily_percentile": target_pct,
                    "model_score": scores,
                }
            )
        dynamic_frame, dynamic_coverage = _dynamic_orderable_frame(
            pair_dir=pair_dir,
            model_frames=dynamic_models,
            model_ids=("MR-12A", "MR-12B", "MR-12C"),
        )
    dynamic_strata = _evaluate_dynamic_k_strata(
        dynamic_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        boundary_width=2,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_dynamic_k_uses_runtime_score_event_date_for_later_occurrences",
        (8, (2,), 2, 1.0, 8, 4, 1.0, 1.0, ("2",), 2),
        (
            int(len(dynamic_frame)),
            tuple(sorted(set(int(value) for value in dynamic_frame["dynamic_k"]))),
            int(dynamic_coverage["full_score_coverage_date_count"]),
            float(dynamic_coverage["full_score_coverage_rate"]),
            int(dynamic_coverage["runtime_score_event_date_row_count"]),
            int(dynamic_coverage["score_event_date_differs_from_signal_date_row_count"]),
            float(dynamic_coverage["common_complete_candidate_row_rate"]),
            float(dynamic_coverage["reference_runtime_scored_candidate_row_rate"]),
            tuple(sorted(dynamic_strata)),
            int(dynamic_strata["2"]["competition_date_count"]),
        ),
    )
    dynamic_strata_table = _render_dynamic_k_strata_table(
        dynamic_strata,
        summary_pair=("MR-12B", "MR-12A"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_dynamic_k_strata_console_renderer_uses_shared_table_contract",
        (True, True, True),
        (
            bool(dynamic_strata_table.strip()),
            "MR-12B NDCG" in dynamic_strata_table,
            "Boundary Δ" in dynamic_strata_table,
        ),
    )
    reference_subset = _evaluate_reference_subset_attribution(
        dynamic_frame,
        dynamic_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
        boundary_width=2,
    )
    reference_subset_table = _render_reference_subset_attribution_table(
        reference_subset,
        summary_pair=("MR-12B", "MR-12A"),
    )
    reference_k2 = dict(reference_subset.get("2") or {})
    reference_event_k2 = dict(reference_k2.get("event_universe") or {})
    reference_orderable_k2 = dict(reference_k2.get("orderable_universe") or {})
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_reference_subset_attribution_holds_dates_and_k_constant_across_universes",
        (("2",), 2, 2, 2, True, True),
        (
            tuple(sorted(reference_subset)),
            int(reference_k2.get("reference_common_complete_date_count", 0)),
            int(reference_event_k2.get("competition_date_count", 0)),
            int(reference_orderable_k2.get("competition_date_count", 0)),
            bool(reference_subset_table.strip()),
            "Orderable NDCG Δ" in reference_subset_table,
        ),
    )

    score_date_frame = pd.DataFrame(
        {
            "date": ["2024-03-10"] * 4,
            "ticker": ["A", "B", "C", "D"],
            "score_event_date": ["2024-03-10", "2024-03-10", "2024-03-01", "2024-03-01"],
            "dynamic_k": [1, 1, 1, 1],
            "target_raw_r": [4.0, 3.0, 2.0, 1.0],
            "score__MR-12A": [0.9, 0.8, 0.7, 0.6],
            "score__MR-12B": [0.2, 0.1, 0.9, 0.8],
            "score__MR-12C": [0.9, 0.8, 0.7, 0.6],
        }
    )
    score_date_diag = _evaluate_score_event_date_comparability(
        score_date_frame,
        model_ids=("MR-12A", "MR-12B", "MR-12C"),
    )
    score_date_table = _render_score_event_date_comparability_table(
        score_date_diag,
        summary_pair=("MR-12B", "MR-12A"),
    )
    same_scope = dict(
        (score_date_diag.get("pair_scopes") or {}).get("same_score_event_date_pairs") or {}
    )
    cross_scope = dict(
        (score_date_diag.get("pair_scopes") or {}).get("cross_score_event_date_pairs") or {}
    )
    same_delta = dict(
        (same_scope.get("contrasts") or {}).get("MR-12B_minus_MR-12A") or {}
    ).get("pairwise_concordance_delta")
    cross_delta = dict(
        (cross_scope.get("contrasts") or {}).get("MR-12B_minus_MR-12A") or {}
    ).get("pairwise_concordance_delta")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_orderable_score_date_comparability_separates_within_date_from_cross_date_pairs",
        (0.5, 1, 1.0, 1.0, 0.0, -1.0, True, True),
        (
            round(float(score_date_diag.get("carried_candidate_row_rate")), 6),
            int(score_date_diag.get("mixed_score_event_date_count", 0)),
            float(score_date_diag.get("mixed_score_event_date_rate")),
            float(score_date_diag.get("k1_mixed_score_event_date_rate")),
            round(float(same_delta), 6),
            round(float(cross_delta), 6),
            "Cross score-date" in score_date_table,
            "K=1 cross score-date" in score_date_table,
        ),
    )

    from filters.breakout_quality.workflow_io import PROJECT_ROOT as breakout_project_root
    missing_trade = _trade_alignment_metrics(
        pd.DataFrame(),
        breakout_project_root / "outputs" / "filters" / "breakout_quality" / "synthetic_missing_trade",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_ranker_missing_trade_reason_uses_project_relative_display_path",
        (False, True, False),
        (
            bool(missing_trade.get("available")),
            str(missing_trade.get("reason", "")).startswith("not found: outputs/"),
            str(breakout_project_root).replace("\\", "/") in str(missing_trade.get("reason", "")),
        ),
    )

    strategy = get_strategy_comparison_settings()
    historical_arm_contract = (
        ("C17", "CONT12A", "resource-aware-continuous-max-dl"),
        ("C18", "CONT12A", "resource-aware-continuous-max-dl-feasible-ascent"),
        ("C19", "CONT12B", "resource-aware-continuous-max-dl"),
        ("C20", "CONT12B", "resource-aware-continuous-max-dl-feasible-ascent"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12a_and_mr12b_historical_c17_c18_selector_matrix_remains_registered",
        historical_arm_contract,
        tuple(
            (
                arm_id,
                strategy.arms.get(arm_id).dl_id if strategy.arms.get(arm_id) is not None else None,
                strategy.arms.get(arm_id).dl_runtime_mode if strategy.arms.get(arm_id) is not None else None,
            )
            for arm_id, _dl_id, _runtime_mode in historical_arm_contract
        ),
    )
    required_contrasts = {
        "C19-C17": ("C19", "C17"),
        "C20-C18": ("C20", "C18"),
        "C20-C19": ("C20", "C19"),
    }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12b_historical_strategy_matrix_is_registered_independent_of_active_config",
        required_contrasts,
        {
            contrast_id: (strategy.contrasts[contrast_id].left, strategy.contrasts[contrast_id].right)
            for contrast_id in required_contrasts
            if contrast_id in strategy.contrasts
        },
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    summary["training_objective"] = TRAINING_OBJECTIVE_DAILY_PAIRWISE_RANKING
    summary["training_performed"] = False
    return results, summary

def validate_breakout_quality_listwise_ranker_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_LISTWISE_RANKER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from filters.breakout_quality.models.factory import require_torch
    from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
    from config.strategy_compare import get_strategy_comparison_settings
    from tools.filters.breakout_quality.train_continuous_ranker import (
        LISTWISE_TRAINING_CONTRACT,
        _date_coherent_batches,
        _profile_contract,
        _listnet_top_one_loss,
        _training_semantics,
        _trade_alignment_metrics,
        parse_args as parse_continuous_ranker_args,
    )

    profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE
    )
    score_eligibility_contract = build_score_eligibility_contract(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12c_profile_changes_pairwise_to_listwise_with_same_target_scope_architecture_family",
        (
            TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING,
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            TRAINING_LABEL_SCOPE_ALL,
            TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
            "listnet_top_one_cross_entropy",
            "mean_daily_spearman",
            False,
        ),
        (
            profile.training_objective,
            profile.continuous_target_id,
            profile.training_label_scope,
            profile.training_sample_scope,
            profile.loss_name,
            profile.epoch_selection_metric,
            STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE
            in SUPPORTED_BREAKOUT_QUALITY_CLASSIFICATION_EXPERIMENT_PROFILES,
        ),
    )

    semantics = _training_semantics(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12c_listwise_artifact_semantics_are_explicit_and_runtime_score_is_unchanged",
        (
            "same_date_full_candidate_list",
            "softmax_daily_percentile",
            "softmax_pass_minus_reject_margin",
            "equal_target_equal_distribution_weight",
            "equal_rankable_date_weight",
            "pass_logit_minus_reject_logit",
            "whole_date_pack_no_date_split",
            "softmax_pass_probability",
        ),
        tuple(
            semantics["listwise_contract"][key]
            for key in (
                "list_scope",
                "target_distribution",
                "prediction_distribution",
                "tie_handling",
                "date_weighting",
                "model_margin",
                "batching",
                "runtime_score",
            )
        ),
    )
    assert semantics["listwise_contract"] == LISTWISE_TRAINING_CONTRACT
    assert semantics["pairwise_contract"] is None

    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.paths import (
        resolve_filter_artifact_paths,
        resolve_filter_model_output_dir,
    )
    from filters.breakout_quality.ranking_score_store import (
        load_continuous_ranker_oos_contract,
        load_continuous_ranker_oos_score_table,
    )
    listwise_loader_accepts = False
    listwise_loader_rejects_corruption = False
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        filter_id = "breakout_quality_v1"
        architecture = "inception_time_v1"
        profile_name = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE
        artifact_paths = resolve_filter_artifact_paths(
            root, filter_id, architecture, profile_name
        )
        output_dir = resolve_filter_model_output_dir(
            root, filter_id, architecture, profile_name
        )
        artifact_paths.model_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_paths.model_path.write_bytes(b"synthetic-model")
        score_path = output_dir / "continuous_ranker_scores.csv"
        pd.DataFrame([
            {
                "ticker": "2330",
                "date": "2021-01-04",
                "group_index": 1,
                "split": "oos",
                "model_score": 0.70,
            }
        ]).to_csv(score_path, index=False)
        report = {
            "status": "RESULT_AVAILABLE",
            "filter_id": filter_id,
            "model_architecture": architecture,
            "experiment_profile": profile_name,
            "training": {
                "objective": profile.training_objective,
                "loss": profile.loss_name,
                "batching": LISTWISE_TRAINING_CONTRACT["batching"],
                "pairwise_contract": None,
                "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
                "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
                "seed": 42,
            },
            "experiment_settings": {
                "continuous_target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            },
            "score_eligibility_contract": score_eligibility_contract,
            "forward_score_coverage": {
                "inference_eligible_groups": 1,
                "target_evaluable_groups": 1,
                "future_target_required_for_score": False,
            },
            "artifacts": {"scores": build_file_manifest(score_path)},
        }
        (output_dir / "continuous_ranker_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        manifest = {
            "filter_id": filter_id,
            "model_architecture": architecture,
            "experiment_profile": profile_name,
            "training_objective": profile.training_objective,
            "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
            "continuous_target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            "training_semantics": {
                "batching": LISTWISE_TRAINING_CONTRACT["batching"],
                "pairwise_contract": None,
                "listwise_contract": dict(LISTWISE_TRAINING_CONTRACT),
            },
            "model": build_file_manifest(artifact_paths.model_path),
            "score_eligibility_contract": score_eligibility_contract,
            "forward_score_coverage": {
                "inference_eligible_groups": 1,
                "target_evaluable_groups": 1,
                "future_target_required_for_score": False,
            },
            "research_outputs": {"scores": build_file_manifest(score_path)},
            "outer_oos_policy": {
                "oos_start_date": "2021-01-01",
                "configured_oos_end_date": "2021-12-31",
            },
            "model_information_cutoff": "2020-12-31",
        }
        artifact_paths.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        load_continuous_ranker_oos_contract.cache_clear()
        load_continuous_ranker_oos_score_table.cache_clear()
        contract = load_continuous_ranker_oos_contract(
            str(root), filter_id, architecture, profile_name
        )
        listwise_loader_accepts = contract.experiment_profile == profile_name
        manifest["training_semantics"]["listwise_contract"] = dict(
            LISTWISE_TRAINING_CONTRACT, tie_handling="invalid"
        )
        artifact_paths.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        load_continuous_ranker_oos_contract.cache_clear()
        try:
            load_continuous_ranker_oos_contract(
                str(root), filter_id, architecture, profile_name
            )
        except ValueError as exc:
            listwise_loader_rejects_corruption = (
                "listwise contract不一致" in str(exc)
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "dl_cont12c_loader_accepts_exact_listwise_contract_and_rejects_semantic_drift",
        (True, True),
        (listwise_loader_accepts, listwise_loader_rejects_corruption),
    )

    args = parse_continuous_ranker_args(
        ["--experiment-profile", STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE]
    )
    contract = _profile_contract(profile)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12c_cli_and_registry_identity_are_explicit",
        (STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE, "12C", "all_labels"),
        (args.experiment_profile, contract["phase"], contract["metric_scope"]),
    )

    dates = pd.Series(["2024-01-02"] * 4 + ["2024-01-03"] * 3 + ["2024-01-04"] * 2)
    batches = _date_coherent_batches(
        np.arange(9, dtype=np.int64), dates, batch_size=5, seed=42
    )
    batch_by_group = {
        int(group_id): int(batch_index)
        for batch_index, batch in enumerate(batches)
        for group_id in batch
    }
    date_batch_counts = []
    for _date, day in pd.DataFrame({"date": dates, "group_id": np.arange(9)}).groupby("date"):
        date_batch_counts.append(len({batch_by_group[int(group_id)] for group_id in day["group_id"]}))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "listwise_training_batches_never_split_same_date_candidate_list",
        (1, 1, 1),
        tuple(date_batch_counts),
    )

    torch, _nn = require_torch()
    targets = torch.tensor([1.0, 1.0, 0.5, 0.0], dtype=torch.float32)
    loss_dates = np.asarray(["2024-01-02"] * 4)
    good_margin = torch.tensor([2.0, 1.5, 0.0, -2.0], dtype=torch.float32, requires_grad=True)
    bad_margin = torch.tensor([-2.0, -1.5, 0.0, 2.0], dtype=torch.float32, requires_grad=True)
    tie_swapped = torch.tensor([1.5, 2.0, 0.0, -2.0], dtype=torch.float32)
    good_loss, good_dates = _listnet_top_one_loss(
        torch, good_margin, targets, loss_dates
    )
    bad_loss, bad_dates = _listnet_top_one_loss(
        torch, bad_margin, targets, loss_dates
    )
    tie_loss, tie_dates = _listnet_top_one_loss(
        torch, tie_swapped, targets, loss_dates
    )
    good_loss.backward()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "listnet_prefers_correct_full_list_distribution_and_is_invariant_within_target_ties",
        (True, 1, 1, 1, True, True),
        (
            bool(float(good_loss.detach().cpu().item()) < float(bad_loss.detach().cpu().item())),
            int(good_dates),
            int(bad_dates),
            int(tie_dates),
            bool(torch.isfinite(good_margin.grad).all().item()),
            bool(torch.isclose(good_loss.detach(), tie_loss.detach(), atol=1e-7).item()),
        ),
    )

    strategy = get_strategy_comparison_settings()
    historical_arm_contract = (
        ("C19", "CONT12B", "resource-aware-continuous-max-dl"),
        ("C20", "CONT12B", "resource-aware-continuous-max-dl-feasible-ascent"),
        ("C21", "CONT12C", "resource-aware-continuous-max-dl"),
        ("C22", "CONT12C", "resource-aware-continuous-max-dl-feasible-ascent"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12b_and_mr12c_historical_selector_matrix_remains_registered",
        historical_arm_contract,
        tuple(
            (
                arm_id,
                strategy.arms.get(arm_id).dl_id if strategy.arms.get(arm_id) is not None else None,
                strategy.arms.get(arm_id).dl_runtime_mode if strategy.arms.get(arm_id) is not None else None,
            )
            for arm_id, _dl_id, _runtime_mode in historical_arm_contract
        ),
    )
    required_contrasts = {
        "C21-C19": ("C21", "C19"),
        "C22-C20": ("C22", "C20"),
        "C22-C21": ("C22", "C21"),
    }
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "mr12c_historical_strategy_matrix_is_registered_independent_of_active_config",
        required_contrasts,
        {
            contrast_id: (strategy.contrasts[contrast_id].left, strategy.contrasts[contrast_id].right)
            for contrast_id in required_contrasts
            if contrast_id in strategy.contrasts
        },
    )

    summary["profile"] = STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_LISTWISE_PROFILE
    summary["training_objective"] = TRAINING_OBJECTIVE_DAILY_LISTWISE_RANKING
    summary["training_performed"] = False
    return results, summary
