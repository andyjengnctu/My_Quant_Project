from __future__ import annotations

from .synthetic_breakout_quality_support import (
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    COMPARISON_MODE_SCORE_RANKING,
    CONTINUOUS_TARGET_SCHEMA_VERSION,
    DEFAULT_LABEL_POLICY,
    LABEL_PASS,
    LABEL_REJECT,
    Path,
    STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    STRATEGY_ALIGNED_TARGET_ID,
    SimpleNamespace,
    StrategyAlignedContinuousTargetSpec,
    TARGET_ADVERSE_RETURN_FILENAME,
    TARGET_FAVORABLE_RETURN_FILENAME,
    TARGET_MANIFEST_FILENAME,
    TARGET_OPPORTUNITY_BAR_FILENAME,
    TARGET_RAW_FILENAME,
    TARGET_RISK_BREACH_BAR_FILENAME,
    TARGET_VALID_MASK_FILENAME,
    V16StrategyParams,
    _candidate_replay_snapshot,
    _registered_breakout_quality_audit_module,
    _source_has_render_menu_item_call,
    add_check,
    approved_no_time_workflow_rebuild_gate,
    assert_qualified_replay_matches_summary,
    ast,
    attach_time_penalty_ablation,
    build_strategy_aligned_no_time_contract,
    build_strategy_aligned_no_time_group_targets,
    json,
    load_validated_continuous_target_arrays,
    load_validated_continuous_target_component_arrays,
    math,
    no_time_target_selection_metrics,
    np,
    os,
    patch,
    pd,
    qualified_audit_actual_trade_metrics,
    qualified_audit_attach_ranker_scores,
    qualified_audit_daily_coverage,
    qualified_audit_layer_metrics,
    qualified_audit_unique_groups,
    render_no_time_target_markdown,
    render_target_attribution_markdown,
    render_time_penalty_ablation_markdown,
    resolve_continuous_target_dir,
    target_attribution_attach_components,
    target_attribution_metrics,
    tempfile,
    time_ablation_validated_source_csv,
    time_penalty_ablation_metrics,
    validate_qualified_audit_strategy_metadata,
    validated_11e_report_for_no_time_target,
)

from .source_index import read_source_ast, read_source_text

def validate_breakout_quality_qualified_candidate_set_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_QUALIFIED_CANDIDATE_SET_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    candidate = {
        "ticker": "2330",
        "trade_date": "2022-01-05",
        "candidate_date": "2022-01-05",
        "signal_date": "2022-01-03",
        "type": "normal",
        "entry_source": "breakout",
        "params_obj": SimpleNamespace(high_len=201),
        "ensemble_vote_count": 4,
        "qty": 1000,
        "sort_value": 0.75,
        "ev": 1.25,
        "hist_win_rate": 0.60,
        "hist_trade_count": 20,
        "breakout_quality_score": 0.63,
        "breakout_quality_score_date": "2022-01-03",
    }
    snapshot = _candidate_replay_snapshot(
        candidate,
        fallback_trade_date=pd.Timestamp("2022-01-05"),
        is_orderable=True,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_snapshot_preserves_original_signal_date_without_runtime_objects",
        (
            "2330", "2022-01-05", "2022-01-03", True, 201, 4, False,
        ),
        (
            snapshot["ticker"], snapshot["trade_date"], snapshot["signal_date"],
            snapshot["is_orderable"], snapshot["high_len"],
            snapshot["ensemble_vote_count"], "params_obj" in snapshot,
        ),
    )

    oos_scores = pd.DataFrame(
        [
            {"ticker": "1101", "date": "2022-01-03", "group_index": 1, "label": 0,
             "target_raw_r": -1.0, "target_daily_percentile": 0.0, "model_score": 0.10},
            {"ticker": "2330", "date": "2022-01-03", "group_index": 2, "label": 1,
             "target_raw_r": 2.0, "target_daily_percentile": 1.0, "model_score": 0.80},
            {"ticker": "2603", "date": "2022-01-04", "group_index": 3, "label": 0,
             "target_raw_r": 0.0, "target_daily_percentile": 0.0, "model_score": 0.20},
            {"ticker": "2454", "date": "2022-01-04", "group_index": 4, "label": 1,
             "target_raw_r": 3.0, "target_daily_percentile": 1.0, "model_score": 0.90},
        ]
    )
    occurrences = pd.DataFrame(
        [
            {"ticker": "1101", "trade_date": "2022-01-04", "candidate_date": "2022-01-04",
             "signal_date": "2022-01-03", "candidate_type": "normal"},
            {"ticker": "1101", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-03", "candidate_type": "continuation"},
            {"ticker": "2330", "trade_date": "2022-01-04", "candidate_date": "2022-01-04",
             "signal_date": "2022-01-03", "candidate_type": "normal"},
            {"ticker": "2603", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
            {"ticker": "2454", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
            {"ticker": "9999", "trade_date": "2022-01-05", "candidate_date": "2022-01-05",
             "signal_date": "2022-01-04", "candidate_type": "normal"},
        ]
    )
    attached = qualified_audit_attach_ranker_scores(
        occurrences,
        oos_scores,
        layer="qualified",
    )
    unique = qualified_audit_unique_groups(attached, layer="qualified")
    metrics = qualified_audit_layer_metrics(
        unique,
        occurrence_count=len(attached),
        occurrence_date_count=int(attached["trade_date"].nunique()),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_occurrences_align_by_signal_date_and_deduplicate_unique_groups",
        (6, 5, 4, 1, 2, 1.0, 1.0),
        (
            len(attached), int(attached["target_match"].sum()), len(unique),
            int((~attached["target_match"]).sum()),
            int(unique.loc[(unique["ticker"] == "1101") & (unique["target_date"] == "2022-01-03"), "occurrence_count"].iloc[0]),
            round(float(metrics["global_spearman_score_vs_target"]), 12),
            round(float(metrics["mean_daily_spearman_score_vs_target"]), 12),
        ),
        tol=1e-12,
    )

    daily = qualified_audit_daily_coverage(
        attached,
        attached[attached["trade_date"] == "2022-01-04"].copy(),
    )
    daily_lookup = daily.set_index("trade_date")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_daily_coverage_counts_ticker_signal_pairs_not_dates_only",
        (2, 2, 4),
        (
            int(daily_lookup.loc["2022-01-04", "qualified_unique_signals"]),
            int(daily_lookup.loc["2022-01-04", "orderable_unique_signals"]),
            int(daily_lookup.loc["2022-01-05", "qualified_unique_signals"]),
        ),
    )

    actual = pd.DataFrame(
        [
            {"ticker": "1101", "target_date": "2022-01-03", "target_raw_r": -1.0, "r_multiple": -0.5},
            {"ticker": "2330", "target_date": "2022-01-03", "target_raw_r": 2.0, "r_multiple": 2.5},
            {"ticker": "2603", "target_date": "2022-01-04", "target_raw_r": 0.0, "r_multiple": 0.0},
            {"ticker": "2454", "target_date": "2022-01-04", "target_raw_r": 3.0, "r_multiple": 4.0},
        ]
    )
    actual_metrics, actual_matches = qualified_audit_actual_trade_metrics(actual, oos_scores)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_actual_trade_merge_preserves_target_score_and_realized_r_direction",
        (4, 4, 1.0, 1.0, 1.0, 4.0, -0.5),
        (
            actual_metrics["trade_count"], actual_metrics["matched_trade_count"],
            round(float(actual_metrics["spearman_target_vs_realized_r"]), 12),
            round(float(actual_metrics["spearman_score_vs_target"]), 12),
            round(float(actual_metrics["spearman_score_vs_realized_r"]), 12),
            float(actual_metrics["top_score_decile_average_r"]),
            float(actual_metrics["bottom_score_decile_average_r"]),
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_actual_trade_output_retains_each_trade_row",
        (4, ("1101", "2330", "2454", "2603")),
        (
            len(actual_matches),
            tuple(sorted(actual_matches["ticker"].astype(str).tolist())),
        ),
    )

    replay_payload = {
        "total_return_pct": 12.5,
        "max_drawdown_pct": 3.0,
        "return_over_max_drawdown": 4.1666666667,
        "annual_return_pct": 2.0,
        "log_r_squared": 0.9,
        "monthly_win_rate_pct": 60.0,
        "trade_count": 4,
        "win_rate_pct": 50.0,
        "payoff_ratio": 2.0,
        "expected_value_r": 0.5,
        "final_equity": 1125000.0,
        "avg_exposure_pct": 40.0,
        "max_exposure_pct": 90.0,
        "missed_buy_count": 1,
        "missed_sell_count": 0,
        "reserved_buy_fill_rate_pct": 80.0,
        "normal_trade_count": 3,
        "extended_trade_count": 1,
        "annual_trade_count": 1.0,
        "benchmark_return_pct": 5.0,
        "benchmark_max_drawdown_pct": 4.0,
        "benchmark_annual_return_pct": 1.0,
        "profile": {
            "portfolio_total_r": 7.5,
            "portfolio_median_r": 0.5,
            "portfolio_avg_r": 1.875,
            "min_full_year_return_pct": -2.0,
            "min_month_return_pct": -1.0,
            "min_quarter_return_pct": -1.5,
            "full_year_count": 1,
            "portfolio_capacity_rows": [],
        },
    }
    expected_replay = {
        key: replay_payload.get(key, replay_payload["profile"].get(key))
        for key in (
            "total_return_pct", "max_drawdown_pct", "return_over_max_drawdown",
            "annual_return_pct", "trade_count", "final_equity",
            "avg_exposure_pct", "max_exposure_pct", "missed_buy_count",
            "missed_sell_count", "normal_trade_count", "extended_trade_count",
            "portfolio_total_r",
        )
    }
    replay_match = assert_qualified_replay_matches_summary(expected_replay, replay_payload)
    bad_replay = dict(replay_payload)
    bad_replay["total_return_pct"] = 99.0
    try:
        assert_qualified_replay_matches_summary(expected_replay, bad_replay)
        replay_mismatch_rejected = False
    except ValueError:
        replay_mismatch_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_capture_must_preserve_formal_no_filter_strategy_results",
        (12.5, 4, 7.5, True),
        (
            float(replay_match["total_return_pct"]),
            int(replay_match["trade_count"]),
            float(replay_match["portfolio_total_r"]),
            replay_mismatch_rejected,
        ),
    )

    valid_metadata = {
        "comparison_mode": "hard-filter",
        "comparison_design": "historical_active_param_oos",
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
        "lookahead_safe_active_param_schedule": True,
        "threshold_used_as_gate": True,
    }
    try:
        validate_qualified_audit_strategy_metadata(
            valid_metadata,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        )
        valid_metadata_accepted = True
    except ValueError:
        valid_metadata_accepted = False
    invalid_metadata_rejected = []
    for mutation in (
        {"comparison_design": "static_param_diagnostic"},
        {"lookahead_safe_active_param_schedule": False},
        {"threshold_used_as_gate": False},
    ):
        payload = dict(valid_metadata)
        payload.update(mutation)
        try:
            validate_qualified_audit_strategy_metadata(
                payload,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            )
            invalid_metadata_rejected.append(False)
        except ValueError:
            invalid_metadata_rejected.append(True)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_accepts_only_lookahead_safe_hard_filter_historical_replay",
        (True, True, True, True),
        (valid_metadata_accepted, *invalid_metadata_rejected),
    )

    app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    app_tree = read_source_ast(app_path)
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    audit_source = (
        Path(__file__).resolve().parents[1]
        / "audit"
        / "breakout_quality"
        / "qualified_candidate_set.py"
    ).read_text(encoding="utf-8")
    strategy_replay_source = (
        Path(__file__).resolve().parents[2]
        / "filters"
        / "breakout_quality"
        / "strategy_compare_replay.py"
    ).read_text(encoding="utf-8")
    engine_source = (
        Path(__file__).resolve().parents[2]
        / "core"
        / "portfolio_engine.py"
    ).read_text(encoding="utf-8")
    runner_source = (
        Path(__file__).resolve().parents[2]
        / "services"
        / "portfolio_replay.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_is_cli_only_and_reuses_canonical_replay",
        (True, True, True, True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-qualified-candidate-set")
            == "tools.audit.breakout_quality.qualified_candidate_set",
            'print("[11] 11C Qualified Candidate-set Audit（research-only）")' not in app_source,
            'elif choice == "11":' not in app_source,
            "run_no_filter_candidate_replay_from_metadata" in audit_source,
            "replay_counts=replay_counts" in strategy_replay_source,
            "replay_counts=None" in engine_source and "replay_counts=None" in runner_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "qualified_candidate_audit_is_diagnostic_only_and_does_not_authorize_training",
        (True, True, True, True, True),
        (
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            '"audit_does_not_authorize_new_model": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )

    summary["command"] = "audit-qualified-candidate-set"
    summary["layers"] = ["all_oos_breakouts", "qualified_candidates", "orderable_candidates", "actual_trades"]
    return results, summary

def validate_breakout_quality_target_component_attribution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TARGET_COMPONENT_ATTRIBUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    target = np.asarray([2.5, 1.5, 0.5, -0.2, -0.7, -1.0], dtype=np.float32)
    favorable = np.asarray([0.25, 0.20, 0.10, 0.08, 0.03, 0.00], dtype=np.float32)
    adverse = np.asarray([0.00, 0.05, 0.05, 0.10, 0.10, 0.10], dtype=np.float32)
    arrays = {
        "target_raw_r": target,
        "valid_mask": np.ones(len(target), dtype=np.bool_),
        "favorable_return": favorable,
        "adverse_return_to_peak": adverse,
        "opportunity_bar": np.ones(len(target), dtype=np.int16),
        "first_risk_breach_bar": np.asarray([-1, -1, -1, -1, -1, 1], dtype=np.int16),
    }
    contract = {
        "risk_budget_return": 0.10,
        "horizon_bars": 40,
        "full_horizon_time_penalty_r": 0.5,
    }
    frame = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(len(target))],
        "target_date": [f"2021-01-{i + 1:02d}" for i in range(len(target))],
        "group_index": np.arange(len(target), dtype=np.int64),
        "label": [LABEL_PASS, LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_REJECT, LABEL_REJECT],
        "target_raw_r": target,
        "model_score": [0.95, 0.85, 0.75, 0.35, 0.25, 0.15],
        "r_multiple": [3.0, 2.0, 1.0, -0.2, -0.8, -1.1],
    })
    attached = target_attribution_attach_components(
        frame,
        arrays=arrays,
        target_contract=contract,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_components_reconstruct_fixed_11a_target",
        True,
        bool(np.allclose(attached["target_reconstructed_r"], target, rtol=0.0, atol=2e-5)),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_component_r_units_are_favorable_minus_adverse_minus_time",
        (2.5, 0.5, -1.0),
        tuple(round(float(value), 6) for value in attached.loc[[0, 2, 5], "target_reconstructed_r"]),
    )

    metrics = target_attribution_metrics(attached, include_realized_r=True)
    correlations = metrics["correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_preserves_score_target_and_target_realized_directions",
        (1.0, 1.0, 1.0),
        (
            round(float(correlations["score_vs_target"]), 6),
            round(float(correlations["target_vs_realized_r"]), 6),
            round(float(correlations["score_vs_realized_r"]), 6),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_reports_pass_and_reject_conditional_metrics",
        (3, 3, 1.0, 1.0),
        (
            int(metrics["by_label"]["pass"]["row_count"]),
            int(metrics["by_label"]["reject"]["row_count"]),
            round(float(metrics["by_label"]["pass"]["correlations"]["target_vs_realized_r"]), 6),
            round(float(metrics["by_label"]["reject"]["correlations"]["target_vs_realized_r"]), 6),
        ),
    )
    markdown = render_target_attribution_markdown({
        "qualified_candidates": target_attribution_metrics(attached, include_realized_r=False),
        "actual_trades": metrics,
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_markdown_exposes_label_conditional_decision_boundary",
        True,
        all(token in markdown for token in ("PASS", "REJECT", "Target↔R", "停止連續排序線")),
    )

    app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    app_tree = read_source_ast(app_path)
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES" for target_node in node.targets):
            command_modules = ast.literal_eval(node.value)
            break
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "audit"
        / "breakout_quality"
        / "target_component_attribution.py"
    )
    audit_source = audit_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_audit_is_cli_only",
        (True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-target-attribution")
            == "tools.audit.breakout_quality.target_component_attribution",
            "11D" not in app_source[app_source.find("def _interactive_model_research"):app_source.find("def run_model_training_menu")],
            'choice == "12"' not in app_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_attribution_audit_is_strict_read_only_failure_attribution",
        (True, True, True, True, True),
        (
            "load_validated_continuous_target_component_arrays" in audit_source,
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = resolve_continuous_target_dir(tmp_dir, "synthetic_filter")
        target_dir.mkdir(parents=True, exist_ok=True)
        artifact_arrays = {
            "target_raw_r": (TARGET_RAW_FILENAME, target),
            "valid_mask": (TARGET_VALID_MASK_FILENAME, np.ones(len(target), dtype=np.bool_)),
            "favorable_return": (TARGET_FAVORABLE_RETURN_FILENAME, favorable),
            "adverse_return_to_peak": (TARGET_ADVERSE_RETURN_FILENAME, adverse),
            "opportunity_bar": (TARGET_OPPORTUNITY_BAR_FILENAME, np.ones(len(target), dtype=np.int16)),
            "first_risk_breach_bar": (
                TARGET_RISK_BREACH_BAR_FILENAME,
                np.asarray([-1, -1, -1, -1, -1, 1], dtype=np.int16),
            ),
        }
        artifacts = {}
        for name, (filename, values) in artifact_arrays.items():
            path = target_dir / filename
            np.save(path, values, allow_pickle=False)
            artifacts[name] = {
                "filename": filename,
                "size_bytes": int(path.stat().st_size),
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": "synthetic_filter",
            "group_count": len(target),
            "dataset_policy": {"synthetic": True},
            "target_contract": {"target_id": STRATEGY_ALIGNED_TARGET_ID},
            "artifacts": artifacts,
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False),
            encoding="utf-8",
        )
        loaded_manifest, loaded_arrays = load_validated_continuous_target_component_arrays(
            tmp_dir,
            "synthetic_filter",
            expected_group_count=len(target),
            expected_dataset_policy={"synthetic": True},
        )
        favorable_path = target_dir / TARGET_FAVORABLE_RETURN_FILENAME
        favorable_path.write_bytes(favorable_path.read_bytes() + b"tamper")
        try:
            load_validated_continuous_target_component_arrays(
                tmp_dir,
                "synthetic_filter",
                expected_group_count=len(target),
                expected_dataset_policy={"synthetic": True},
            )
            tamper_rejected = False
        except ValueError:
            tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_component_loader_validates_all_artifacts_and_rejects_tamper",
        (True, True, True),
        (
            str((loaded_manifest.get("target_contract") or {}).get("target_id")) == STRATEGY_ALIGNED_TARGET_ID,
            bool(np.allclose(loaded_arrays["favorable_return"], favorable)),
            tamper_rejected,
        ),
    )
    summary["command"] = "audit-target-attribution"
    summary["layers"] = ["qualified_candidates", "actual_trades", "pass", "reject"]
    return results, summary

def validate_breakout_quality_target_time_penalty_ablation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TARGET_TIME_PENALTY_ABLATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    favorable = np.asarray([3.2, 2.7, 2.2, 1.7, 1.2, 0.7], dtype=np.float64)
    adverse = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
    time_penalty = np.asarray([0.0, 1.8, 0.0, 0.8, 0.0, 0.0], dtype=np.float64)
    no_time = favorable - adverse
    original = no_time - time_penalty
    frame = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(6)],
        "target_date": [f"2021-02-{i + 1:02d}" for i in range(6)],
        "group_index": np.arange(6, dtype=np.int64),
        "label": [LABEL_PASS, LABEL_PASS, LABEL_PASS, LABEL_REJECT, LABEL_REJECT, LABEL_REJECT],
        "model_score": original,
        "target_raw_r": original,
        "favorable_r": favorable,
        "adverse_r": adverse,
        "time_penalty_r": time_penalty,
        "r_multiple": no_time,
    })
    attached = attach_time_penalty_ablation(frame)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_derives_fixed_favorable_minus_adverse_target",
        tuple(round(float(value), 6) for value in no_time),
        tuple(round(float(value), 6) for value in attached["target_no_time_r"]),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reconstructs_original_target_exactly",
        True,
        bool(np.allclose(
            attached["target_original_reconstructed_r"],
            attached["target_raw_r"],
            rtol=0.0,
            atol=2e-5,
        )),
    )

    qualified_metrics = time_penalty_ablation_metrics(
        attached,
        include_realized_r=False,
    )
    actual_metrics = time_penalty_ablation_metrics(
        attached,
        include_realized_r=True,
    )
    corr = actual_metrics["correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reports_original_and_no_time_economic_directions",
        (1.0, True),
        (
            round(float(corr["no_time_target_vs_realized_r"]), 6),
            float(corr["no_time_target_vs_realized_r"])
            > float(corr["original_target_vs_realized_r"]),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_reports_positive_delta_and_decile_spread",
        (True, True),
        (
            float(actual_metrics["deltas"]["spearman_no_time_minus_original"]) > 0.0,
            float(actual_metrics["deltas"]["decile_spread_no_time_minus_original"]) >= 0.0,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_preserves_pass_reject_conditional_rows",
        (3, 3),
        (
            int(actual_metrics["by_label"]["pass"]["row_count"]),
            int(actual_metrics["by_label"]["reject"]["row_count"]),
        ),
    )

    markdown = render_time_penalty_ablation_markdown({
        "qualified_candidates": qualified_metrics,
        "actual_trades": actual_metrics,
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_markdown_exposes_single_change_and_training_boundary",
        True,
        all(token in markdown for token in (
            "target_no_time_r = favorable_r - adverse_r",
            "PASS",
            "REJECT",
            "不授權訓練",
            "不測time penalty反向加分",
        )),
    )

    app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    app_tree = read_source_ast(app_path)
    command_modules = {}
    for node in app_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "audit"
        / "breakout_quality"
        / "target_time_penalty_ablation.py"
    )
    audit_source = audit_path.read_text(encoding="utf-8")
    menu_source = app_source[
        app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_audit_is_cli_only",
        (True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-target-time-ablation")
            == "tools.audit.breakout_quality.target_time_penalty_ablation",
            "11E" not in menu_source,
            "audit-target-time-ablation" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_is_strict_single_read_only_ablation",
        (True, True, True, True, True, True),
        (
            "target_no_time_r = favorable_r - adverse_r" in audit_source,
            '"single_fixed_ablation": "remove_time_penalty_only"' in audit_source,
            '"training_performed": False' in audit_source,
            '"research_only": True' in audit_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "source.csv"
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        report = {
            "artifacts": {
                "synthetic": {
                    "sha256": __import__("hashlib").sha256(csv_path.read_bytes()).hexdigest(),
                }
            }
        }
        accepted = time_ablation_validated_source_csv(
            report,
            key="synthetic",
            canonical_path=csv_path,
        ) == csv_path
        csv_path.write_text("tampered", encoding="utf-8")
        try:
            time_ablation_validated_source_csv(
                report,
                key="synthetic",
                canonical_path=csv_path,
            )
            tamper_rejected = False
        except ValueError:
            tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "time_ablation_strictly_validates_11d_source_hash",
        (True, True),
        (accepted, tamper_rejected),
    )
    summary["command"] = "audit-target-time-ablation"
    summary["ablation"] = "remove_time_penalty_only"
    return results, summary

def validate_breakout_quality_no_time_target_selection_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_NO_TIME_TARGET_SELECTION_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    source_contract = StrategyAlignedContinuousTargetSpec.from_label_policy(
        DEFAULT_LABEL_POLICY
    ).contract_payload()
    contract = build_strategy_aligned_no_time_contract(source_contract)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_contract_is_versioned_fixed_and_selection_only",
        (
            STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            STRATEGY_ALIGNED_TARGET_ID,
            False,
            True,
        ),
        (
            contract.get("target_id"),
            contract.get("source_target_id"),
            contract.get("time_penalty_included"),
            contract.get("current_audit_oos_rows_evaluated") is False,
        ),
    )

    approved_settings = SimpleNamespace(
        is_continuous_ranker=True,
        filter_id="synthetic",
        experiment_profile="strategy_aligned_no_time_pass_magnitude_mse",
        continuous_target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
    )
    with patch(
        "tools.audit.breakout_quality.no_time_continuous_target.get_breakout_quality_workflow_settings",
        return_value=approved_settings,
    ):
        approved_gate = approved_no_time_workflow_rebuild_gate(filter_id="synthetic")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_active_workflow_rebuild_uses_fixed_formula_without_rechecking_history",
        ("active_workflow_profile", False, True),
        (
            approved_gate.get("approval_basis"),
            approved_gate.get("historical_research_gate_recomputed"),
            approved_gate.get("fixed_formula_only"),
        ),
    )

    favorable_return = np.asarray([0.30, 0.20, 0.10, np.nan], dtype=np.float32)
    adverse_return = np.asarray([0.02, 0.04, 0.01, np.nan], dtype=np.float32)
    opportunity = np.asarray([2, 4, 1, -1], dtype=np.int16)
    risk_breach = np.asarray([-1, 5, 1, -1], dtype=np.int16)
    valid = np.asarray([True, True, True, False], dtype=bool)
    arrays = build_strategy_aligned_no_time_group_targets(
        favorable_return=favorable_return,
        adverse_return_to_peak=adverse_return,
        opportunity_bar=opportunity,
        first_risk_breach_bar=risk_breach,
        valid_mask=valid,
        risk_budget_return=0.10,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_arrays_equal_favorable_minus_adverse_in_r_units",
        (2.8, 1.6, 0.9, True),
        (
            round(float(arrays["target_raw_r"][0]), 6),
            round(float(arrays["target_raw_r"][1]), 6),
            round(float(arrays["target_raw_r"][2]), 6),
            bool(np.isnan(arrays["target_raw_r"][3])),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        target_dir = resolve_continuous_target_dir(
            tmp_dir,
            "synthetic",
            target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / TARGET_RAW_FILENAME
        valid_path = target_dir / TARGET_VALID_MASK_FILENAME
        np.save(target_path, arrays["target_raw_r"], allow_pickle=False)
        np.save(valid_path, arrays["valid_mask"], allow_pickle=False)
        sha = __import__("hashlib").sha256
        manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": "synthetic",
            "target_contract": contract,
            "group_count": 4,
            "dataset_policy": {"synthetic": True},
            "artifacts": {
                "target_raw_r": {
                    "filename": target_path.name,
                    "size_bytes": target_path.stat().st_size,
                    "sha256": sha(target_path.read_bytes()).hexdigest(),
                },
                "valid_mask": {
                    "filename": valid_path.name,
                    "size_bytes": valid_path.stat().st_size,
                    "sha256": sha(valid_path.read_bytes()).hexdigest(),
                },
            },
        }
        (target_dir / TARGET_MANIFEST_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        loaded_manifest, loaded_target, loaded_valid = load_validated_continuous_target_arrays(
            tmp_dir,
            "synthetic",
            target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
            expected_group_count=4,
            expected_dataset_policy={"synthetic": True},
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_version_is_loadable_by_strict_public_loader",
        (STRATEGY_ALIGNED_NO_TIME_TARGET_ID, True, True),
        (
            (loaded_manifest.get("target_contract") or {}).get("target_id"),
            bool(np.allclose(loaded_target[:3], arrays["target_raw_r"][:3])),
            bool(np.array_equal(loaded_valid, arrays["valid_mask"])),
        ),
    )

    frame = pd.DataFrame({
        "date": pd.to_datetime([
            "2018-01-02", "2018-01-02", "2018-01-03", "2018-01-03",
            "2020-01-02", "2020-01-02",
        ]),
        "label": [LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT, LABEL_PASS, LABEL_REJECT],
        "target_raw_r": [2.8, 0.5, 1.8, -0.2, 2.2, 0.1],
        "source_target_raw_r": [2.5, 0.4, 1.2, -0.3, 1.8, 0.0],
        "max_upside_return": [0.30, 0.08, 0.22, 0.02, 0.26, 0.04],
        "decision_mfe_return": [0.28, 0.07, 0.20, 0.01, 0.24, 0.03],
        "decision_mae_return": [-0.02, -0.05, -0.03, -0.08, -0.02, -0.07],
        "valid_mask": [True] * 6,
        "is_inner_train": [True, True, True, True, False, False],
        "is_validation": [False, False, False, False, True, True],
        "is_selection": [True] * 6,
    })
    metrics, daily = no_time_target_selection_metrics(frame)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_selection_metrics_cover_only_three_selection_splits",
        (("inner_train", "selection", "validation"), 6, True),
        (
            tuple(sorted(metrics)),
            int(metrics["selection"]["group_count"]),
            set(daily["split"].astype(str)) == {"inner_train", "validation", "selection"},
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_selection_metrics_preserve_rankability_and_source_comparison",
        (1.0, 1.0, True),
        (
            round(float(metrics["validation"]["same_day_rankability"]["rankable_date_rate"]), 6),
            round(float(metrics["selection"]["same_day_rankability"]["pairwise_non_tie_rate"]), 6),
            float(metrics["selection"]["source_vs_no_time_spearman"]) > 0.0,
        ),
    )

    markdown = render_no_time_target_markdown({
        "target_contract": contract,
        "split_metrics": metrics,
        "source_11e_gate": {
            "overall_spearman_delta": 0.0859,
            "pass_spearman_delta": 0.0942,
            "decile_spread_delta": 0.6029,
        },
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_markdown_exposes_selection_only_and_iterative_oos_boundary",
        True,
        all(token in markdown for token in (
            "Selection-only",
            "OOS邊界",
            "不建立OOS指標",
            "不訓練",
            "prior" if False else "迭代OOS",
        )),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        ranker_dir = Path(tmp_dir)
        audit_dir = ranker_dir / "target_time_penalty_ablation_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        qualified_path = audit_dir / "qualified_time_penalty_ablation.csv"
        actual_path = audit_dir / "actual_trade_time_penalty_ablation.csv"
        pd.DataFrame({"x": [1]}).to_csv(qualified_path, index=False, encoding="utf-8-sig")
        pd.DataFrame({"x": [2]}).to_csv(actual_path, index=False, encoding="utf-8-sig")
        sha = __import__("hashlib").sha256
        report = {
            "status": "RESULT_AVAILABLE_PENDING_REVIEW",
            "source_continuous_target_id": STRATEGY_ALIGNED_TARGET_ID,
            "interpretation_contract": {
                "research_only": True,
                "training_performed": False,
            },
            "artifacts": {
                "qualified_ablation": {"sha256": sha(qualified_path.read_bytes()).hexdigest()},
                "actual_trade_ablation": {"sha256": sha(actual_path.read_bytes()).hexdigest()},
            },
            "actual_trades": {
                "correlations": {
                    "original_target_vs_realized_r": 0.40,
                    "no_time_target_vs_realized_r": 0.49,
                },
                "deltas": {
                    "spearman_no_time_minus_original": 0.09,
                    "decile_spread_no_time_minus_original": 0.60,
                },
                "by_label": {
                    "pass": {
                        "correlations": {
                            "original_target_vs_realized_r": 0.36,
                            "no_time_target_vs_realized_r": 0.45,
                        }
                    }
                },
            },
        }
        report_path = audit_dir / "target_time_penalty_ablation_audit.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with patch(
            "tools.audit.breakout_quality.no_time_continuous_target._ranker_dir",
            return_value=ranker_dir,
        ):
            accepted, accepted_path = validated_11e_report_for_no_time_target(
                filter_id="synthetic",
                ranker_profile=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
            )
            accepted_ok = accepted_path == report_path and accepted.get("status") == report["status"]
            actual_path.write_text("tampered", encoding="utf-8")
            try:
                validated_11e_report_for_no_time_target(
                    filter_id="synthetic",
                    ranker_profile=STRATEGY_ALIGNED_DAILY_PERCENTILE_MSE_PROFILE,
                )
                tamper_rejected = False
            except ValueError:
                tamper_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_requires_positive_11e_gate_and_strict_artifact_hashes",
        (True, True),
        (accepted_ok, tamper_rejected),
    )

    app_path = Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    app_tree = read_source_ast(app_path)
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")
    ]
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "audit"
        / "breakout_quality"
        / "no_time_continuous_target.py"
    )
    audit_source = audit_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_audit_is_cli_only_and_read_only",
        (True, True, True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-no-time-target")
            == "tools.audit.breakout_quality.no_time_continuous_target",
            "11F" not in menu_source,
            "audit-no-time-target" not in menu_source,
            "torch.save(" not in audit_source,
            "optimizer" not in audit_source.lower(),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "no_time_target_audit_does_not_compute_oos_metrics",
        (True, True, True),
        (
            'SELECTION_SPLITS = ("inner_train", "validation", "selection")' in audit_source,
            '"oos_evaluated": False' in audit_source,
            '"oos_rows_scores_labels_or_statistics_evaluated": False' in audit_source,
        ),
    )

    summary["command"] = "audit-no-time-target"
    summary["target_id"] = STRATEGY_ALIGNED_NO_TIME_TARGET_ID
    return results, summary










def validate_breakout_quality_minimum_repair_mechanism_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_MINIMUM_REPAIR_MECHANISM_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config.audit import get_audit_definitions
    from tools.audit.catalog import get_audit_entry
    from tools.audit.breakout_quality.minimum_repair_mechanism import (
        _conclusion,
        collect_minimum_repair_mechanism_status,
    )

    definitions = {item.audit_id: item for item in get_audit_definitions("breakout_quality")}
    definition = definitions["mr13e-minimum-repair-mechanism"]
    entry = get_audit_entry("minimum_repair_mechanism")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_mechanism_audit_is_active_config_driven_read_only_scalable_search_certificate_step",
        True,
        bool(
            len(definitions) == 1
            and definition.enabled
            and definition.audit_type == "minimum_repair_mechanism"
            and definition.dimensions.get("exact_one_swap_certificate") is True
            and definition.dimensions.get("multi_step_path_unresolved") is True
            and definition.outcomes.get("frozen_score_only_search_certificate") is True
            and definition.outcomes.get("no_numeric_threshold") is True
            and entry.formal
            and entry.read_only
            and entry.module == "tools.audit.breakout_quality.minimum_repair_mechanism"
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocked = collect_minimum_repair_mechanism_status(definition, project_root=Path(tmp))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_mechanism_audit_blocks_cleanly_without_search_certificate_sidecars",
        True,
        blocked.get("status") == "BLOCKED",
    )

    def arm(*, one_step, multi_step, fallback, one_contrib, multi_contrib, fallback_contrib):
        return {
            "status": "AVAILABLE",
            "classification_counts": {
                "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT": int(one_step),
                "MULTI_STEP_GREEDY_PATH_UNRESOLVED": int(multi_step),
                "BASELINE_FALLBACK_AFTER_GREEDY_REPAIR": int(fallback),
            },
            "class_target_delta_contribution": {
                "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT": float(one_contrib),
                "MULTI_STEP_GREEDY_PATH_UNRESOLVED": float(multi_contrib),
                "BASELINE_FALLBACK_AFTER_GREEDY_REPAIR": float(fallback_contrib),
            },
        }

    payload = {
        "phases": {
            "forward_oos": {
                "arms": {
                    "C20": arm(one_step=4, multi_step=2, fallback=0, one_contrib=-0.10, multi_contrib=-0.05, fallback_contrib=0.0),
                    "C29": arm(one_step=3, multi_step=3, fallback=0, one_contrib=-0.15, multi_contrib=-0.20, fallback_contrib=0.0),
                    "C36": arm(one_step=5, multi_step=2, fallback=0, one_contrib=-0.40, multi_contrib=-0.10, fallback_contrib=0.0),
                }
            }
        }
    }
    conclusion = _conclusion(payload)
    extra = dict(conclusion.get("mr13e_minus_mr12b_class_contribution_r") or {})
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_conclusion_separates_exact_one_swap_resource_days_from_multi_step_unresolved_contribution_without_threshold",
        True,
        bool(
            conclusion.get("classification") == "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT_DOMINATES"
            and conclusion.get("dominant_class") == "EXACT_ONE_SWAP_RESOURCE_CONSTRAINT"
            and math.isclose(
                float(extra.get("EXACT_ONE_SWAP_RESOURCE_CONSTRAINT")),
                -0.30,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(extra.get("MULTI_STEP_GREEDY_PATH_UNRESOLVED")),
                -0.05,
                abs_tol=1e-12,
            )
        ),
    )

    summary["workflow"] = "mr13e_minimum_repair_mechanism_audit"
    return results, summary

def validate_breakout_quality_audit_framework_contract_case(_base_params):
    from config.audit import (
        AUDIT_OUTPUT_ROOT,
        get_audit_definitions,
        get_audit_module_ids,
        get_enabled_audit_definitions,
        validate_audit_config,
    )
    from tools.audit.catalog import get_audit_entry, validate_audit_catalog
    from tools.audit.runner import collect_audit_status

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validate_audit_config()
    definitions = get_audit_definitions("breakout_quality")
    enabled = get_enabled_audit_definitions("breakout_quality")
    validate_audit_catalog(definitions)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_config_contains_only_current_decision_relevant_definition",
        True,
        bool(
            get_audit_module_ids() == ("breakout_quality",)
            and len(definitions) == 1
            and len(enabled) == 1
            and definitions[0].audit_type == "minimum_repair_mechanism"
            and AUDIT_OUTPUT_ROOT == "outputs/audit"
        ),
    )
    entry = get_audit_entry(definitions[0].audit_type)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "current_formal_audit_catalog_entry_is_read_only_and_loadable",
        True,
        bool(
            entry.formal
            and entry.read_only
            and callable(entry.load_status_handler())
            and callable(entry.load_run_handler())
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        status_payload = collect_audit_status(
            module_id="breakout_quality",
            project_root=Path(tmp),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "formal_audit_runner_reports_current_audit_without_historical_handlers",
        True,
        bool(
            len(status_payload.get("rows") or []) == 1
            and status_payload["rows"][0].get("audit_id") == definitions[0].audit_id
            and status_payload["rows"][0].get("status") == "BLOCKED"
        ),
    )

    summary["workflow"] = "current_minimal_audit_framework"
    return results, summary
