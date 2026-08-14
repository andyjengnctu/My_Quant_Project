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

def validate_breakout_quality_pass_realization_gap_attribution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PASS_REALIZATION_GAP_ATTRIBUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.pass_realization_gap import (
        attach_no_time_components,
        partial_spearman,
        realization_gap_metrics,
        render_markdown,
    )

    rng = np.random.default_rng(10)
    count = 30
    score = np.arange(count, dtype=np.float64) / float(count - 1)
    latent = rng.normal(size=count)
    target = 0.5 * score + latent
    target = target - float(target.min()) + 0.6
    realized = 1.2 * latent - 2.5 * score + rng.normal(scale=0.1, size=count)
    adverse = 0.1 + 0.2 * np.linspace(0.0, 1.0, count)
    favorable = target + adverse
    frame = pd.DataFrame({
        "group_index": np.arange(count, dtype=np.int64),
        "label": np.full(count, LABEL_PASS, dtype=np.int64),
        "target_raw_r": target,
        "model_score": score,
        "r_multiple": realized,
    })
    arrays = {
        "target_raw_r": target.astype(np.float32),
        "favorable_return": favorable.astype(np.float32) * 0.1,
        "adverse_return_to_peak": adverse.astype(np.float32) * 0.1,
        "valid_mask": np.ones(count, dtype=bool),
    }
    attached = attach_no_time_components(
        frame,
        arrays=arrays,
        risk_budget_return=0.1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_components_reconstruct_no_time_target",
        (True, True, True),
        (
            bool(np.allclose(attached["favorable_r"] - attached["adverse_r"], attached["target_raw_r"], atol=1e-5, rtol=0.0)),
            bool((attached["label"] == LABEL_PASS).all()),
            bool(np.isfinite(attached[["target_raw_r", "favorable_r", "adverse_r", "model_score"]].to_numpy(dtype=np.float64)).all()),
        ),
    )

    metrics = realization_gap_metrics(attached, include_realized_r=True)
    corr = metrics["correlations"]
    partial = metrics["partial_correlations"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_detects_target_learning_but_negative_realized_r",
        (True, True, True, True),
        (
            float(corr["score_vs_target"]) > 0.20,
            float(corr["target_vs_realized_r"]) > 0.45,
            float(corr["score_vs_realized_r"]) < -0.10,
            float(corr["score_vs_realization_gap_r"]) > 0.80,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_partial_rank_controls_target",
        (True, True),
        (
            float(partial["score_vs_realized_r_controlling_target"]) < -0.50,
            partial_spearman(score, realized, [target]) is not None,
        ),
    )
    score_deciles = metrics["score_deciles"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_score_deciles_expose_unrealized_opportunity",
        (True, True, True),
        (
            float(score_deciles["top"]["mean_target_raw_r"]) > float(score_deciles["bottom"]["mean_target_raw_r"]),
            float(score_deciles["top"]["mean_realization_gap_r"]) > float(score_deciles["bottom"]["mean_realization_gap_r"]),
            float(score_deciles["top"]["mean_r_multiple"]) < float(score_deciles["bottom"]["mean_r_multiple"]),
        ),
    )

    markdown = render_markdown({
        "oos_pass": realization_gap_metrics(attached.drop(columns=["r_multiple"]), include_realized_r=False),
        "actual_pass": {key: value for key, value in metrics.items() if key != "frame"},
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_report_exposes_gap_capture_partial_and_boundary",
        (True, True, True, True),
        (
            "Target−R gap" in markdown,
            "R÷Favorable" in markdown,
            "Partial Score↔R" in markdown,
            "strategy-realization target audit" in markdown,
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    tree = read_source_ast(app_path)
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[
        app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")
    ]
    audit_path = root / "tools" / "audit" / "breakout_quality" / "pass_realization_gap.py"
    audit_source = audit_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_is_cli_only_and_registered",
        (True, True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-pass-realization-gap")
            == "tools.audit.breakout_quality.pass_realization_gap",
            "11H" not in menu_source,
            "audit-pass-realization-gap" not in menu_source,
            "STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE" in audit_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_strictly_validates_sources",
        (True, True, True, True),
        (
            "11H偵測到11G scores SHA256不一致" in audit_source,
            "11H偵測到11A trade matches SHA256不一致" in audit_source,
            "No-time target無法由favorable-adverse逐筆重建" in audit_source,
            "actual PASS配對數與11G report不一致" in audit_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pass_realization_gap_audit_is_read_only_and_does_not_authorize_model",
        (True, True, True, True),
        (
            '"research_only": True' in audit_source,
            '"training_performed": False' in audit_source,
            '"new_model_not_authorized_until_result_review": True' in audit_source,
            "torch.save" not in audit_source,
        ),
    )

    summary["ranker_profile"] = STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    summary["audit"] = "pass_realization_gap"
    return results, summary

def validate_breakout_quality_selection_strategy_realization_contract_case(_base_params):
    from config.training_policy import OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT
    from filters.breakout_quality.trade_path_label import (
        TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        TRADE_PATH_SELECTION_BASELINE_TRAIN_WINDOW_MONTHS,
    )
    from tools.audit.breakout_quality.selection_strategy_realization import (
        DEFAULT_NESTED_OOS_END_DATE,
        DEFAULT_REPLAY_END_DATE,
        DEFAULT_START_DATE,
        DEFAULT_TRAIN_WINDOW_MONTHS,
        parse_args as parse_selection_strategy_realization_args,
    )

    case_id = "BREAKOUT_QUALITY_SELECTION_STRATEGY_REALIZATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.selection_strategy_realization import (
        _attach_targets,
        _trade_metrics,
        _unique_signals,
        _validate_replay_target_bounds,
        _write_prepare_script,
        default_params_path,
        default_research_models_dir,
    )

    lookup = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "target_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "group_index": [0, 1, 2],
        "label": [LABEL_PASS, LABEL_PASS, LABEL_REJECT],
        "target_raw_r": [2.0, 0.5, -0.2],
        "target_valid": [True, True, True],
    })
    candidates = pd.DataFrame({
        "ticker": ["A", "A", "B", "C"],
        "trade_date": ["2018-01-03", "2018-01-04", "2018-01-04", "2018-01-05"],
        "candidate_date": ["2018-01-02", "2018-01-02", "2018-01-03", "2018-01-04"],
        "signal_date": ["2018-01-02", "2018-01-02", "2018-01-03", "2018-01-04"],
        "candidate_type": ["normal", "extended", "normal", "normal"],
    })
    attached = _attach_targets(candidates, lookup)
    unique = _unique_signals(attached)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_candidate_mapping_uses_signal_date_and_dedup",
        (4, 3, [0, 1, 2], True),
        (
            len(attached),
            len(unique),
            sorted(unique["group_index"].astype(int).tolist()),
            bool(unique["target_match"].all()),
        ),
    )

    trades = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "entry_date": ["2018-01-03", "2018-01-04", "2018-01-05"],
        "signal_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "candidate_date": ["2018-01-02", "2018-01-03", "2018-01-04"],
        "r_multiple": [2.5, 0.2, -0.5],
    })
    trade_attached = _attach_targets(trades, lookup)
    metrics = _trade_metrics(trade_attached)
    target_manifest = {
        "split_report": {
            "final_refit_date_range": {"start": "2011-01-03", "end": "2020-11-05"}
        }
    }
    valid_bounds = _validate_replay_target_bounds(
        target_manifest,
        start_date="2014-01-01",
        end_date="2020-11-05",
    )
    try:
        _validate_replay_target_bounds(
            target_manifest,
            start_date="2014-01-01",
            end_date="2020-12-31",
        )
        embargo_rejected = False
    except ValueError:
        embargo_rejected = True
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_replay_respects_selection_target_boundary",
        (("2011-01-03", "2020-11-05"), True),
        (valid_bounds, embargo_rejected),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_metrics_preserve_target_and_r_direction",
        (3, 3, True, 2, 1),
        (
            metrics["trade_count"],
            metrics["matched_trade_count"],
            float(metrics["spearman_target_vs_realized_r"]) > 0.5,
            metrics["label_conditional"]["PASS"]["rows"],
            metrics["label_conditional"]["REJECT"]["rows"],
        ),
    )

    default_args = parse_selection_strategy_realization_args([])
    override_args = parse_selection_strategy_realization_args(["--optimizer-trials", "17"])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_trial_default_uses_training_policy_single_source",
        (int(OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT), 17),
        (int(default_args.optimizer_trials), int(override_args.optimizer_trials)),
    )

    research_dir = default_research_models_dir()
    params_path = default_params_path()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_uses_isolated_research_models_dir",
        (True, True, "roos_base_finalists_agree.json"),
        (
            "models" in research_dir.parts and "research" in research_dir.parts,
            params_path.parent == research_dir,
            params_path.name,
        ),
    )

    root = Path(__file__).resolve().parents[2]
    app_path = root / "tools" / "filters" / "breakout_quality" / "application.py"
    app_source = read_source_text(app_path)
    tree = read_source_ast(app_path)
    command_modules = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target_node, ast.Name) and target_node.id == "COMMAND_MODULES"
            for target_node in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")]
    audit_path = root / "tools" / "audit" / "breakout_quality" / "selection_strategy_realization.py"
    audit_source = audit_path.read_text(encoding="utf-8")
    optimizer_path = root / "services" / "optimizer" / "outer_rolling_oos.py"
    optimizer_source = optimizer_path.read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_is_cli_only_and_registered",
        (True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-selection-strategy-realization")
            == "tools.audit.breakout_quality.selection_strategy_realization",
            "audit-selection-strategy-realization" not in menu_source,
            "11I" not in menu_source,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_prepare_contract_is_lookahead_safe_and_isolated",
        (True, True, True, True, True, True),
        (
            DEFAULT_START_DATE == TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
            pd.Timestamp(DEFAULT_START_DATE)
            <= pd.Timestamp(DEFAULT_REPLAY_END_DATE)
            < pd.Timestamp(DEFAULT_NESTED_OOS_END_DATE),
            DEFAULT_NESTED_OOS_END_DATE == TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
            (
                "--outer-train-window-months" in audit_source
                and DEFAULT_TRAIN_WINDOW_MONTHS
                == TRADE_PATH_SELECTION_BASELINE_TRAIN_WINDOW_MONTHS
            ),
            "V16_MODELS_DIR" in audit_source,
            "resolve_models_dir(project_root, environ=environ)" in optimizer_source,
        ),
    )
    from tools.optimizer.outer_rolling_oos import _resolve_config as _resolve_outer_rolling_config

    option_only_argv = [
        "--outer-first-oos-date",
        str(DEFAULT_START_DATE),
        "--outer-last-oos-date",
        str(DEFAULT_NESTED_OOS_END_DATE),
        "--outer-train-window-months",
        str(int(DEFAULT_TRAIN_WINDOW_MONTHS)),
        "--outer-oos-months",
        "12",
        "--trials",
        "17",
    ]
    parser_policy = {
        # AI註: 故意與Selection期間不同，第一個argv token若再被漏讀就必須失敗。
        "oos_start_year": 2021,
        "oos_end_date": "2026-03-02",
    }
    option_only_config = _resolve_outer_rolling_config(
        option_only_argv,
        {},
        base_policy=parser_policy,
        latest_year=2026,
        latest_date=pd.Timestamp("2026-03-02"),
        default_trials=300,
        timing_mode=False,
    )
    sys_argv_config = _resolve_outer_rolling_config(
        ["tools/optimizer/main.py", *option_only_argv],
        {},
        base_policy=parser_policy,
        latest_year=2026,
        latest_date=pd.Timestamp("2026-03-02"),
        default_trials=300,
        timing_mode=False,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "outer_rolling_parser_accepts_cli_and_programmatic_argv_without_dropping_first_option",
        (2014, 2020, 120, 12, 17, True),
        (
            int(option_only_config.first_oos_year),
            int(option_only_config.last_oos_year),
            int(option_only_config.train_window_months),
            int(option_only_config.oos_horizon_months),
            int(option_only_config.trials_per_fold),
            option_only_config == sys_argv_config,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_nested_roos_trial_default_has_no_duplicate_magic_number",
        (True, True),
        (
            "default=OPTIMIZER_OUTER_ROLLING_OOS_TRIALS_DEFAULT" in audit_source,
            'parser.add_argument("--optimizer-trials", type=int, default=1000)' not in audit_source,
        ),
    )
    from tools.optimizer.outer_rolling_oos import OuterRollingConfig, _write_reports
    with tempfile.TemporaryDirectory() as td:
        temp_root = Path(td)
        prepare_path = _write_prepare_script(
            output_dir=temp_root / "prepare",
            trials=17,
            dataset_profile="reduced",
        )
        prepare_source = prepare_path.read_text(encoding="utf-8-sig")
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_nested_roos_prepare_script_preserves_dataset_and_cleans_env",
            (True, True, True, True),
            (
                "--dataset reduced" in prepare_source,
                "--trials 17" in prepare_source,
                f"--outer-last-oos-date {DEFAULT_NESTED_OOS_END_DATE}" in prepare_source,
                "Remove-Item Env:V16_MODELS_DIR" in prepare_source,
            ),
        )
        isolated_models = temp_root / "isolated_models"
        write_result = _write_reports(
            project_root=str(temp_root),
            output_dir=str(temp_root / "outputs"),
            session_ts="synthetic",
            rows=[],
            config=OuterRollingConfig(
                training_start_year=2004,
                first_oos_year=2014,
                last_oos_year=2020,
                trials_per_fold=1,
                train_window_months=120,
                oos_horizon_months=12,
            ),
            models_dir=str(isolated_models),
        )
        written_paths = [Path(path) for path in (write_result.get("paramsets") or {}).values()]
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_nested_roos_writer_behaves_as_isolated_override",
            (True, True, False),
            (
                bool(written_paths),
                bool(written_paths) and all(path.parent == isolated_models for path in written_paths),
                (temp_root / "models").exists(),
            ),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_strategy_realization_does_not_label_untraded_candidates_or_authorize_training",
        (True, True, True, True),
        (
            "untraded_candidates_are_not_labeled_zero" in audit_source,
            '"training_performed": False' in audit_source,
            '"runtime_eligible": False' in audit_source,
            "torch.save" not in audit_source,
        ),
    )

    summary["audit"] = "selection_strategy_realization"
    return results, summary

def validate_breakout_quality_candidate_counterfactual_execution_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_CANDIDATE_COUNTERFACTUAL_EXECUTION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.portfolio_engine import _candidate_execution_replay_snapshot
    from core.strategy_params import V16StrategyParams
    from tools.audit.breakout_quality.candidate_counterfactual_execution import (
        CandidateCounterfactualReplay,
        _execution_market_dates,
        _metrics as counterfactual_metrics,
        _render_markdown as render_counterfactual_markdown,
        _run_offline_counterfactual,
    )

    dates = tuple(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    fast = {
        "_packed_market_data": True,
        "dates": dates,
        "date_to_pos": {date: idx for idx, date in enumerate(dates)},
        "security_profile": None,
        "Open": np.asarray([99.0, 99.0, 89.0]),
        "High": np.asarray([100.0, 103.0, 90.0]),
        "Low": np.asarray([98.0, 98.0, 88.0]),
        "Close": np.asarray([99.0, 102.0, 89.0]),
        "Volume": np.asarray([1000.0, 1000.0, 1000.0]),
        "ATR": np.asarray([5.0, 5.0, 5.0]),
        "buy_limit": np.asarray([100.0, 100.0, 100.0]),
        "is_setup": np.asarray([False, False, False]),
        "ind_sell_signal": np.asarray([False, False, False]),
    }
    params = V16StrategyParams()
    candidate = {
        "ticker": "2330",
        "signal_date": "2020-01-01",
        "candidate_date": "2020-01-02",
        "trade_date": dates[1],
        "type": "normal",
        "entry_source": "normal",
        "qty": 1000,
        "limit_px": 100.0,
        "init_sl": 90.0,
        "init_trail": 90.0,
        "target_price": 110.0,
        "entry_atr": None,
        "security_profile": None,
        "today_pos": 1,
        "yesterday_pos": 0,
        "sizing_capital": 1_000_000.0,
        "params_obj": params,
        "is_orderable": True,
    }
    snapshot = {
        "ticker": "2330",
        "trade_date": "2020-01-02",
        "candidate_date": "2020-01-02",
        "signal_date": "2020-01-01",
    }

    tracker = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02")
    tracker.begin_replay_day(today=dates[1], all_dfs_fast={"2330": fast}, fallback_params=params)
    tracker.observe_replay_candidates(
        today=dates[1],
        qualified_candidates=[candidate],
        qualified_candidate_snapshots=[snapshot],
        orderable_candidates=[candidate],
        orderable_candidate_snapshots=[snapshot],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
        fallback_params=params,
    )
    tracker.begin_replay_day(today=dates[2], all_dfs_fast={"2330": fast}, fallback_params=params)
    frame = tracker.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_reuses_canonical_entry_and_exit_for_filled_signal",
        (1, True, True, "2020-01-02", "2020-01-03", "STOP", True),
        (
            len(frame), bool(frame.iloc[0]["filled"]), bool(frame.iloc[0]["closed"]),
            frame.iloc[0]["entry_date"], frame.iloc[0]["exit_date"], frame.iloc[0]["exit_type"],
            float(frame.iloc[0]["r_multiple"]) < 0.0,
        ),
    )

    large_params_payload = {"weights": list(range(10000))}
    shadow = {"qty": 1000, "sl": 90.0, "_last_exec_contexts": [{"x": 1}]}
    memory_candidate = dict(candidate)
    memory_candidate["params_obj"] = large_params_payload
    memory_candidate["signal_state"] = {
        "_params_obj": large_params_payload,
        "shadow_position": shadow,
    }
    sidecar = _candidate_execution_replay_snapshot(
        memory_candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
    )
    shadow["qty"] = 1
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_execution_sidecar_preserves_read_only_refs_and_clones_shadow_only",
        (True, True, True, True, True, "2020-01-01"),
        (
            sidecar.get("params_obj") is large_params_payload,
            "signal_state" not in sidecar,
            sidecar.get("_candidate_fast_df") is fast,
            int((sidecar.get("shadow_position_state") or {}).get("qty", 0)) == 1000,
            (sidecar.get("shadow_position_state") or {}).get("_last_exec_contexts") is not shadow.get("_last_exec_contexts"),
            (sidecar.get("_canonical_snapshot") or {}).get("signal_date"),
        ),
    )

    execution_row = _candidate_execution_replay_snapshot(
        candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
    )
    canonical_qualified = pd.DataFrame([{
        "ticker": "2330",
        "target_date": "2020-01-01",
        "trade_date": "2020-01-02",
        "candidate_type": "normal",
        "entry_source": "normal",
    }])
    offline = _run_offline_counterfactual(
        execution_rows=[execution_row],
        canonical_qualified=canonical_qualified,
        candidate_cutoff="2020-01-02",
        replay_end="2020-01-03",
    )
    offline_frame = offline.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_executes_sidecar_only_after_canonical_replay",
        (True, True, False, 1),
        (
            bool(offline_frame.iloc[0]["filled"]),
            bool(offline_frame.iloc[0]["closed"]),
            isinstance(offline, dict),
            int(offline.offline_management_day_count),
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_market_calendar_is_derived_from_sidecar_fast_data",
        tuple(dates),
        tuple(_execution_market_dates([execution_row], start_date="2020-01-01", end_date="2020-01-03")),
    )

    canonical_tracker = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02")
    raw_a = dict(candidate, signal_date="", candidate_date="2020-01-02")
    raw_b = dict(candidate, signal_date="", candidate_date="2020-01-02")
    snapshot_a = dict(snapshot, signal_date="2020-01-01")
    snapshot_b = dict(snapshot, signal_date="2020-01-02")
    canonical_tracker.observe_replay_candidates(
        today=dates[1],
        qualified_candidates=[raw_a, raw_b],
        qualified_candidate_snapshots=[snapshot_a, snapshot_b],
        orderable_candidates=[],
        orderable_candidate_snapshots=[],
        all_dfs_fast={"2330": fast},
        sizing_equity=1_000_000.0,
        fallback_params=params,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_uses_canonical_snapshot_keys",
        (("2330", "2020-01-01"), ("2330", "2020-01-02")),
        tuple(sorted(canonical_tracker.states)),
    )

    mismatch_rejected = False
    try:
        canonical_tracker.observe_replay_candidates(
            today=dates[1],
            qualified_candidates=[raw_a, raw_b],
            qualified_candidate_snapshots=[snapshot_a],
            orderable_candidates=[],
            orderable_candidate_snapshots=[],
            all_dfs_fast={"2330": fast},
            sizing_equity=1_000_000.0,
            fallback_params=params,
        )
    except ValueError as exc:
        mismatch_rejected = "canonical snapshot數量不一致" in str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_rejects_snapshot_length_divergence",
        True,
        mismatch_rejected,
    )

    deferred = CandidateCounterfactualReplay(candidate_cutoff="2020-01-02", defer_finalize=True)
    deferred.observe_replay_candidates(
        today=dates[1], qualified_candidates=[candidate], qualified_candidate_snapshots=[snapshot],
        orderable_candidates=[candidate], orderable_candidate_snapshots=[snapshot],
        all_dfs_fast={"2330": fast}, sizing_equity=1_000_000.0, fallback_params=params,
    )
    deferred.finalize_replay(last_date=dates[1], fallback_params=params)
    discovery_frame = deferred.signal_frame()
    deferred.defer_finalize = False
    deferred.finalize_replay(last_date=dates[2], fallback_params=params)
    managed_frame = deferred.signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_defers_closeout_until_execution_end",
        (True, False, True, "FORCED_CLOSE"),
        (
            bool(discovery_frame.iloc[0]["filled"]),
            bool(discovery_frame.iloc[0]["closed"]),
            bool(managed_frame.iloc[0]["closed"]),
            managed_frame.iloc[0]["exit_type"],
        ),
    )

    high_open_fast = dict(fast)
    high_open_fast.update({
        "Open": np.asarray([110.0, 110.0, 110.0]),
        "High": np.asarray([111.0, 111.0, 111.0]),
        "Low": np.asarray([109.0, 109.0, 109.0]),
        "Close": np.asarray([110.0, 110.0, 110.0]),
    })
    unfilled_candidate = dict(candidate)
    unfilled_row = _candidate_execution_replay_snapshot(
        unfilled_candidate,
        fallback_trade_date=dates[1],
        all_dfs_fast={"2330": high_open_fast},
        sizing_equity=1_000_000.0,
    )
    unfilled = _run_offline_counterfactual(
        execution_rows=[unfilled_row],
        canonical_qualified=canonical_qualified,
        candidate_cutoff="2020-01-02",
        replay_end="2020-01-03",
    ).signal_frame()
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_keeps_unfilled_signal_nan_not_zero",
        (False, False, True, 1),
        (
            bool(unfilled.iloc[0]["filled"]), bool(unfilled.iloc[0]["closed"]),
            bool(pd.isna(unfilled.iloc[0]["r_multiple"])),
            int(unfilled.iloc[0]["missed_buy_count"]),
        ),
    )

    metric_frame = pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "target_match": [True, True, True],
        "was_orderable": [True, True, False],
        "filled": [True, False, False],
        "closed": [True, False, False],
        "target_raw_r": [2.0, 1.0, 0.0],
        "r_multiple": [1.5, np.nan, np.nan],
        "label": [LABEL_PASS, LABEL_PASS, LABEL_REJECT],
    })
    metrics = counterfactual_metrics(metric_frame)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_coverage_denominators_remain_explicit",
        (3, 2, 1, 1/3, 1/2, 1/3),
        (
            metrics["qualified_signal_count"], metrics["orderable_signal_count"], metrics["filled_signal_count"],
            metrics["fill_coverage_vs_qualified"], metrics["fill_coverage_vs_orderable"],
            metrics["strategy_r_coverage_vs_qualified"],
        ),
        tol=1e-12,
    )
    markdown = render_counterfactual_markdown({
        "metrics": metrics,
        "source_11i": {"actual_trade_coverage_vs_qualified": 0.22},
    })
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_report_states_capacity_and_unfilled_boundary",
        True,
        all(token in markdown for token in (
            "忽略portfolio capacity", "未成交候選維持unlabeled", "不建立Target arrays", "不授權模型",
        )),
    )

    root = Path(__file__).resolve().parents[2]
    app_source = (root / "tools" / "filters" / "breakout_quality" / "application.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")]
    engine_source = (root / "core" / "portfolio_engine.py").read_text(encoding="utf-8")
    runner_source = (root / "services" / "portfolio_replay.py").read_text(encoding="utf-8")
    compare_replay_source = (
        root / "filters" / "breakout_quality" / "strategy_compare_replay.py"
    ).read_text(encoding="utf-8")
    entry_source = (root / "core" / "portfolio_entries.py").read_text(encoding="utf-8")
    audit_source = (
        root / "tools" / "audit" / "breakout_quality" / "candidate_counterfactual_execution.py"
    ).read_text(encoding="utf-8")
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_cli_only_and_sidecar_is_not_replay_counts",
        (True, True, True, True, True, True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-candidate-counterfactual")
            == "tools.audit.breakout_quality.candidate_counterfactual_execution",
            "11J" not in menu_source,
            "audit-candidate-counterfactual" not in menu_source,
            'getattr(replay_counts, "begin_replay_day", None)' not in engine_source,
            'getattr(replay_counts, "observe_replay_candidates", None)' not in engine_source,
            "replay_execution_rows=None" in engine_source,
            "replay_execution_rows=replay_execution_rows" in runner_source,
            "replay_execution_rows=replay_execution_rows" in compare_replay_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_main_uses_plain_dict_counts_then_sidecar",
        (True, True, True, True, True, True),
        (
            "discovery_counts: dict[str, dict[str, Any]] = {}" in audit_source,
            "execution_rows: list[dict[str, Any]] = []" in audit_source,
            "replay_counts=discovery_counts" in audit_source,
            "replay_execution_rows=execution_rows" in audit_source,
            '"replay_counts_type":"plain_dict"' in audit_source,
            '"lifecycle_callbacks_used":False' in audit_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_reuses_entry_shadow_exit_and_accounting_ssot",
        (True, True, True, True, True),
        (
            "build_candidate_plan_seed" in entry_source,
            "execute_pre_market_entry_plan" in audit_source,
            "execute_bar_step" in audit_source,
            "closeout_open_positions" in audit_source,
            "calc_ratio_from_milli" in audit_source,
        ),
    )
    discovery_call = audit_source.find('name="11J_candidate_discovery"')
    source_count_guard = audit_source.find("canonical_discovery_count!=source_qualified_count", discovery_call)
    offline_call = audit_source.find("tracker=_run_offline_counterfactual(", source_count_guard)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_checks_2003_before_offline_execution",
        (True, True, True, True),
        (
            discovery_call >= 0,
            source_count_guard > discovery_call,
            offline_call > source_count_guard,
            '"execution_mode":"plain_replay_counts_with_execution_sidecar"' in audit_source,
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "candidate_counterfactual_is_read_only_and_requires_11i_positive_result",
        (True, True, True, True, True, True),
        (
            '"training_performed":False' in audit_source,
            '"runtime_eligible":False' in audit_source,
            "11J需要11I overall與PASS Target↔R均為正" in audit_source,
            "11J偵測到11I artifact SHA256不一致" in audit_source,
            "copy.deepcopy" not in audit_source,
            "torch.save" not in audit_source,
        ),
    )
    summary["command"] = "audit-candidate-counterfactual"
    summary["audit"] = "canonical_per_candidate_execution"
    return results, summary

def validate_breakout_quality_portfolio_selection_pressure_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_PORTFOLIO_SELECTION_PRESSURE"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from tools.audit.breakout_quality.portfolio_selection_pressure import (
        _build_selection_pressure_tables,
        _render_markdown as render_selection_pressure_markdown,
        parse_args as parse_selection_pressure_args,
    )

    orderable = pd.DataFrame([
        {"ticker": "A", "trade_date": "2020-01-02", "target_raw_r": 3.0, "target_match": True},
        {"ticker": "B", "trade_date": "2020-01-02", "target_raw_r": 2.0, "target_match": True},
        {"ticker": "C", "trade_date": "2020-01-02", "target_raw_r": 1.0, "target_match": True},
        {"ticker": "D", "trade_date": "2020-01-03", "target_raw_r": 4.0, "target_match": True},
        {"ticker": "E", "trade_date": "2020-01-03", "target_raw_r": 1.0, "target_match": True},
    ])
    trades = pd.DataFrame([
        {"ticker": "A", "entry_date": "2020-01-02", "target_raw_r": 3.0, "target_match": True, "r_multiple": 2.0},
        {"ticker": "E", "entry_date": "2020-01-03", "target_raw_r": 1.0, "target_match": True, "r_multiple": -1.0},
    ])
    signals, daily, buckets, metrics = _build_selection_pressure_tables(orderable, trades)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_same_day_percentile_and_top_k_are_exact",
        (5, 2, 0.75, 0.5, 1.5),
        (
            len(signals), metrics["selected_trade_count"],
            metrics["selected_target_percentile_mean_competition"],
            metrics["top_k_retention_occurrence_weighted"],
            metrics["target_opportunity_gap_r_date_weighted"],
        ),
        tol=1e-12,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_preserves_realized_r_only_for_selected_rows",
        (2, 3, True),
        (
            int(signals["r_multiple"].notna().sum()),
            int(signals["r_multiple"].isna().sum()),
            bool(signals.loc[~signals["selected"], "r_multiple"].isna().all()),
        ),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_daily_and_bucket_outputs_are_explicit",
        (2, ["1", "2-3", "4-5", "6-10", "11+"], True),
        (
            len(daily), buckets["pressure_bucket"].tolist(),
            bool({"top_k_retention", "target_opportunity_gap_r"}.issubset(buckets.columns)),
        ),
    )

    duplicate_rejected = False
    try:
        _build_selection_pressure_tables(pd.concat([orderable, orderable.iloc[[0]]], ignore_index=True), trades)
    except ValueError as exc:
        duplicate_rejected = "同ticker／trade_date存在多筆" in str(exc)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_rejects_duplicate_candidate_day_identity",
        True,
        duplicate_rejected,
    )

    markdown = render_selection_pressure_markdown({"metrics": metrics})
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_report_states_non_counterfactual_boundary",
        True,
        all(token in markdown for token in (
            "不重播市場", "未交易候選沒有realized R", "不填0R", "不授權使用future Target作runtime排序",
        )),
    )

    root = Path(__file__).resolve().parents[2]
    app_source = (root / "tools" / "filters" / "breakout_quality" / "application.py").read_text(encoding="utf-8")
    app_tree = ast.parse(app_source)
    command_modules = {}
    for node in app_tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "COMMAND_MODULES" for target in node.targets
        ):
            command_modules = ast.literal_eval(node.value)
            break
    menu_source = app_source[app_source.find("def _interactive_model_research") : app_source.find("def run_model_training_menu")]
    audit_source = (
        root / "tools" / "audit" / "breakout_quality" / "portfolio_selection_pressure.py"
    ).read_text(encoding="utf-8")
    args = parse_selection_pressure_args([])
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "portfolio_selection_pressure_is_cli_only_read_only_and_uses_11i_artifacts",
        (True, True, True, True, True, True, True, True),
        (
            _registered_breakout_quality_audit_module("audit-selection-pressure")
            == "tools.audit.breakout_quality.portfolio_selection_pressure",
            "11K" not in menu_source,
            "audit-selection-pressure" not in menu_source,
            args.filter_id == BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            '"strategy_replay_performed": False' in audit_source,
            '"counterfactual_performed": False' in audit_source,
            '"training_performed": False' in audit_source,
            "_artifact_path(source_dir, source, \"orderable\")" in audit_source,
        ),
    )

    summary["command"] = "audit-selection-pressure"
    summary["audit"] = "portfolio_selection_pressure"
    return results, summary

def validate_breakout_quality_score_ranking_capture_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SCORE_RANKING_CAPTURE_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from core.console_report import (
        project_relative_display_path,
        strip_ansi,
    )
    from tools.audit.portfolio.score_ranking_capture import (
        _aggregate_capture_ratio,
        _decision,
        build_score_ranking_capture_audit,
        render_capture_audit_console,
        write_score_ranking_capture_audit_outputs,
    )
    from filters.breakout_quality.strategy_compare_engine import (
        COMPARISON_MODE_SCORE_RANKING,
        _markdown_report,
        _render_strategy_console_report,
    )

    baseline_history = pd.DataFrame([
        {
            "Date": "2014-01-02", "Ticker": "2330",
            "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2014-01-01",
            "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0,
            "停損價": 90.0, "股數": 1000, "預留總金額": 101000.0,
            "投入總金額": 100500.0, "Quality Score": 0.6,
            "Quality Score Date": "2014-01-01",
        },
        {"Date": "2014-01-10", "Ticker": "2330", "Type": "半倉停利", "成交價": 115.0, "股數": 500},
        {"Date": "2014-01-20", "Ticker": "2330", "Type": "全倉結算(指標)", "該筆總損益": 12000.0, "R_Multiple": 1.2},
        {"Date": "2014-02-01", "Ticker": "1101", "Type": "錯失買進(新訊號)", "預留總金額": 50000.0},
    ])
    score_history = pd.DataFrame([
        {
            "Date": "2014-01-02", "Ticker": "2330",
            "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2014-01-01",
            "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0,
            "停損價": 88.0, "股數": 800, "預留總金額": 100000.0,
            "投入總金額": 80500.0, "Quality Score": 0.8,
            "Quality Score Date": "2014-01-01",
        },
        {"Date": "2014-01-25", "Ticker": "2330", "Type": "全倉結算(停損)", "該筆總損益": -7000.0, "R_Multiple": -0.7},
        {"Date": "2014-02-01", "Ticker": "1101", "Type": "錯失買進(新訊號)", "預留總金額": 50000.0},
    ])
    selected = pd.DataFrame([
        {
            "ticker": "2330", "trade_date": "2014-01-02",
            "signal_date": "2014-01-01", "score_event_date": "2014-01-01",
            "target_raw_r": 2.0,
        }
    ])
    baseline_capacity = pd.DataFrame({
        "Date": [
            "2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07",
            "2014-01-08", "2014-01-09", "2014-01-10", "2014-01-13",
            "2014-01-14", "2014-01-15", "2014-01-16", "2014-01-17",
            "2014-01-20", "2014-01-25", "2014-02-01",
        ]
    })
    score_capacity = baseline_capacity.copy()
    metadata = {
        "comparison_mode": COMPARISON_MODE_SCORE_RANKING,
        "comparison_period": {"start": "2014-01-01", "end": "2014-12-31"},
        "score_source": "selection_point_in_time",
        "params_path": "synthetic.json", "param_source_kind": "rolling_active_param_ensemble",
        "param_selector": "base_finalist_best", "runtime_member_count_min": 1,
        "runtime_member_count_max": 1, "runtime_min_agree": 1,
        "comparison_design": "selection_point_in_time_active_param_replay",
        "lookahead_safe_active_param_schedule": True, "dataset": "full",
        "benchmark_ticker": "0050", "filter_id": "synthetic",
        "model_architecture": "inception_time_v1", "experiment_profile": "synthetic",
        "score_ranking_order": ["breakout_quality_score_desc", "existing_buy_sort", "ticker_deterministic"],
        "ranking_scope": "all_candidates_after_single_member_qualification",
    }
    baseline_summary = {
        "total_return_pct": 20.0, "max_drawdown_pct": 10.0,
        "return_over_max_drawdown": 2.0, "annual_return_pct": 10.0,
        "log_r_squared": 0.9, "monthly_win_rate_pct": 60.0, "trade_count": 1,
        "win_rate_pct": 100.0, "payoff_ratio": 2.0, "expected_value_r": 1.2,
        "avg_exposure_pct": 77.0, "min_full_year_return_pct": 20.0,
        "avg_orderable_candidates": 10.0, "candidate_supply_gap_days": 1,
        "underfilled_end_days": 1, "end_position_gap_slot_days": 1,
    }
    score_summary = {
        **baseline_summary,
        "total_return_pct": 10.0, "max_drawdown_pct": 20.0,
        "return_over_max_drawdown": 0.5, "avg_exposure_pct": 55.0,
        "expected_value_r": -0.7, "win_rate_pct": 0.0,
    }
    selection_diagnostics = {
        "no_filter": {}, "score_ranking": {},
        "score_ranking_minus_no_filter": {
            "selected_target_mean_r": 0.2,
            "target_top_k_retention_mean": 0.1,
        },
    }
    result = build_score_ranking_capture_audit(
        metadata=metadata,
        baseline_summary=baseline_summary,
        score_sort_summary=score_summary,
        baseline_trade_history=baseline_history,
        score_sort_trade_history=score_history,
        baseline_selected_target_diagnostics=selected,
        score_sort_selected_target_diagnostics=selected,
        selection_diagnostics=selection_diagnostics,
        baseline_daily_capacity=baseline_capacity,
        score_sort_daily_capacity=score_capacity,
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capture_audit_reconstructs_capital_partial_and_target_capture_metrics",
        (100500.0, 80500.0, 6, 0.6, -0.35, 0.6, -0.35, 100.0, 100.0, 0.0, None),
        (
            result["baseline"]["avg_invested_total"],
            result["score_sort"]["avg_invested_total"],
            result["baseline"]["partial_residual_slot_days"],
            result["baseline"]["aggregate_target_capture_ratio"],
            result["score_sort"]["aggregate_target_capture_ratio"],
            result["baseline"]["raw_mean_target_capture_ratio"],
            result["score_sort"]["raw_mean_target_capture_ratio"],
            result["baseline"]["top_5_entry_dates_share_pct"],
            result["baseline"]["top_entry_month_share_pct"],
            result["baseline"]["industry_coverage_pct"],
            result["baseline"]["top_industry_share_pct"],
        ),
    )
    aggregate_all_targets = _aggregate_capture_ratio(pd.DataFrame({
        "target_raw_r": [2.0, -1.0],
        "r_multiple": [1.0, -0.25],
    }))
    aggregate_target_ge_0_5 = _aggregate_capture_ratio(pd.DataFrame({
        "target_raw_r": [2.0, -1.0],
        "r_multiple": [1.0, -0.25],
    }), min_target_r=0.5)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "aggregate_capture_uses_sum_realized_over_sum_target_and_threshold_variant",
        (0.75, 0.5),
        (aggregate_all_targets, aggregate_target_ge_0_5),
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capture_audit_supports_parameter_adaptation_only_after_target_improves_and_sort_only_fails",
        ("ADAPTATION_DIAGNOSTIC_SUPPORTED", True, True, True, True, True, False),
        (
            result["decision"]["status"],
            result["decision"]["target_selection_improved"],
            result["decision"]["economic_effect_failed"],
            result["decision"]["parameter_adaptation_candidate"],
            result["decision"]["mechanical_bottleneck_detected"],
            result["decision"]["realization_gap_detected"],
            result["decision"]["future_target_used_for_runtime"],
        ),
    )

    capture_only_decision = _decision(
        baseline={
            "avg_invested_total": 100000.0,
        },
        score_sort={
            "avg_invested_total": 98000.0,
        },
        delta={
            "total_return_pct": -10.0,
            "return_over_max_drawdown": -0.5,
            "avg_exposure_pct": -0.2,
            "aggregate_target_capture_ratio": -0.30,
            "median_target_capture_ratio": -0.20,
            "target_ge_0_5_capture_ratio": -0.25,
            "avg_target_realization_gap_r": 0.15,
            "avg_holding_calendar_days": 0.5,
            "avg_partial_to_exit_calendar_days": 0.0,
            "reserved_buy_fill_rate_pct": -1.0,
        },
        selection_diagnostics={
            "score_ranking_minus_no_filter": {
                "selected_target_mean_r": 0.05,
                "target_top_k_retention_mean": 0.01,
            }
        },
    )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "capture_gap_alone_is_not_parameter_adaptation_mechanical_evidence",
        (
            "SORT_ONLY_REJECTED_REALIZATION_GAP_NO_MECHANICAL_BOTTLENECK",
            True,
            True,
            False,
            False,
            True,
        ),
        (
            capture_only_decision["status"],
            capture_only_decision["target_selection_improved"],
            capture_only_decision["economic_effect_failed"],
            capture_only_decision["parameter_adaptation_candidate"],
            capture_only_decision["mechanical_bottleneck_detected"],
            capture_only_decision["realization_gap_detected"],
        ),
    )

    yearly = pd.DataFrame([
        {
            "year": 2014, "no_filter_return_pct": 20.0,
            "score_ranking_return_pct": 10.0, "delta_pct": -10.0,
            "is_full_year": True,
        }
    ])
    comparison_delta = {
        key: float(score_summary[key]) - float(baseline_summary[key])
        for key in score_summary
        if isinstance(score_summary.get(key), (int, float))
        and isinstance(baseline_summary.get(key), (int, float))
    }
    comparison_markdown = _markdown_report(
        metadata, baseline_summary, score_summary, comparison_delta, yearly, selection_diagnostics
    )
    comparison_console = _render_strategy_console_report(
        metadata, baseline_summary, score_summary, comparison_delta, yearly,
        selection_diagnostics, color=True,
    )
    capture_console = render_capture_audit_console(result, color=True)
    with patch.dict(os.environ, {"BREAKOUT_QUALITY_COMPACT_CONSOLE": "1"}):
        compact_comparison_console = _render_strategy_console_report(
            metadata, baseline_summary, score_summary, comparison_delta, yearly,
            selection_diagnostics, color=True,
        )
        compact_capture_console = render_capture_audit_console(result, color=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        (output_dir / "score_ranking_capture_audit.html").write_text(
            "legacy", encoding="utf-8"
        )
        payload = write_score_ranking_capture_audit_outputs(result=result, output_dir=temp_dir)
        audit_markdown = (output_dir / "score_ranking_capture_audit.md").read_text(encoding="utf-8")
        outputs_complete = all(
            (output_dir / name).is_file()
            for name in (
                "score_ranking_capture_audit.json", "score_ranking_capture_audit.md",
                "no_filter_capture_lifecycle.csv", "score_ranking_capture_lifecycle.csv",
                "score_ranking_capture_yearly.csv", "score_ranking_capture_scenarios.csv",
            )
        )
        no_html_outputs = not (output_dir / "score_ranking_capture_audit.html").exists()
    compact_strategy_text = strip_ansi(compact_comparison_console)
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_compare_and_audit_are_separate_read_only_reports",
        (True, True, True),
        (
            "Breakout Quality" in compact_strategy_text,
            "Target Capture" not in compact_strategy_text,
            "Score Sort 資金配置與 Target Capture 診斷"
            in strip_ansi(compact_capture_console),
        ),
    )

    add_check(
        results, "synthetic_breakout_quality", case_id,
        "strategy_and_capture_reports_are_console_readable_without_html_outputs",
        (True, True, True, True, True),
        (
            outputs_complete,
            no_html_outputs,
            (
                "🔴 惡化" in comparison_markdown
                and "🟡 注意" in audit_markdown
                and "半倉殘留交易slot-days" in audit_markdown
                and "產業資料覆蓋" in audit_markdown
            ),
            (
                "Breakout Quality Score 排序策略經濟效果對照" in strip_ansi(comparison_console)
                and "Breakout Quality Score 排序資本效率／Target Capture 歸因" in strip_ansi(capture_console)
                and "Exit reason" in strip_ansi(capture_console)
                and "\x1b[" in comparison_console
                and "\x1b[" in capture_console
            ),
            payload["decision"]["status"] == "ADAPTATION_DIAGNOSTIC_SUPPORTED",
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        relative_root = Path(temp_dir)
        relative_output = relative_root / "outputs" / "filters" / "breakout_quality" / "report.json"
        relative_output.parent.mkdir(parents=True, exist_ok=True)
        relative_output.write_text("{}", encoding="utf-8")
        displayed_path = project_relative_display_path(
            relative_output,
            project_root=relative_root,
        )
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "console_artifact_paths_are_project_root_relative_and_forward_slashed",
        "outputs/filters/breakout_quality/report.json",
        displayed_path,
    )

    summary["workflow"] = "score_ranking_capture_audit"
    summary["future_target_runtime"] = False
    return results, summary

def validate_breakout_quality_orderable_feasible_alignment_audit_contract_case(_base_params):
    from config.audit import get_audit_definitions
    from tools.audit.breakout_quality import orderable_feasible_alignment as alignment_audit
    from tools.audit.breakout_quality import orderable_alignment_common as alignment_common
    from tools.audit.breakout_quality.orderable_feasible_alignment import (
        _capacity_frame,
        _daily_ranking_rows,
        _execution_translation,
        _information_date_map,
        _load_common_daily_target,
    )
    from tools.audit.catalog import get_audit_entry
    from tools.audit.sources.strategy_compare import resolve_strategy_compare_profile_run_selector

    case_id = "BREAKOUT_QUALITY_ORDERABLE_FEASIBLE_ALIGNMENT_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    definitions = get_audit_definitions("breakout_quality")
    definition = next(
        (item for item in definitions if item.audit_id == "mr13e-orderable-feasible-alignment"),
        None,
    )
    cross_period = next(
        (item for item in definitions if item.audit_id == "cross-period-year-regime-attribution"),
        None,
    )
    selector_stage = next(
        (item for item in definitions if item.audit_id == "mr13e-selector-stage-translation"),
        None,
    )
    minimum_repair = next(
        (item for item in definitions if item.audit_id == "mr13e-minimum-repair-mechanism"),
        None,
    )
    entry = get_audit_entry("orderable_feasible_alignment")
    phases = {} if definition is None else dict(dict(definition.source).get("phases") or {})
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_config_is_current_cross_phase_source_only_topology_without_fixed_k",
        True,
        bool(
            definition is not None
            and not definition.enabled
            and definition.audit_type == "orderable_feasible_alignment"
            and dict(definition.source).get("kind") == "strategy_compare_cross_phase"
            and tuple(dict(phases.get("selection_pit") or {}).get("candidate_arm_ids") or ()) == ("C25", "C28", "C35")
            and tuple(dict(phases.get("forward_oos") or {}).get("candidate_arm_ids") or ()) == ("C20", "C29", "C36")
            and dict(phases.get("selection_pit") or {}).get("reference_daily_arm_id") == "C35"
            and dict(phases.get("forward_oos") or {}).get("reference_daily_arm_id") == "C36"
            and definition.dimensions.get("dynamic_action_count") is True
            and definition.outcomes.get("no_fixed_k") is True
            and cross_period is not None
            and not cross_period.enabled
            and selector_stage is not None
            and not selector_stage.enabled
            and minimum_repair is not None
            and minimum_repair.enabled
        ),
    )
    status_handler = entry.load_status_handler()
    run_handler = entry.load_run_handler()
    with tempfile.TemporaryDirectory() as tmp:
        blocked_status = status_handler(definition, project_root=Path(tmp))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_catalog_is_formal_read_only_importable_and_blocks_cleanly_without_source_artifacts",
        True,
        bool(
            entry.formal
            and entry.read_only
            and entry.module == "tools.audit.breakout_quality.orderable_feasible_alignment"
            and callable(status_handler)
            and callable(run_handler)
            and callable(resolve_strategy_compare_profile_run_selector)
            and blocked_status.get("status") == "BLOCKED"
        ),
    )

    from config.strategy_compare import get_strategy_comparison_settings
    with tempfile.TemporaryDirectory() as tmp:
        synthetic_root = Path(tmp).resolve()
        compare_settings = get_strategy_comparison_settings("selection_pit")
        synthetic_run = synthetic_root / compare_settings.output_root / "runs" / "synthetic_orderable_alignment"
        synthetic_run.mkdir(parents=True, exist_ok=True)
        synthetic_result = {
            "status": "COMPLETED",
            "config_fingerprint": "synthetic-alignment",
            "settings": {
                "arms": {arm_id: compare_settings.arms[arm_id].as_dict() for arm_id in ("C23", "C25", "C28", "C35")},
            },
            "scenarios": {arm_id: {} for arm_id in ("C23", "C25", "C28", "C35")},
        }
        (synthetic_run / "strategy_comparison.json").write_text(
            json.dumps(synthetic_result), encoding="utf-8"
        )
        latest_dir = synthetic_root / compare_settings.output_root / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": synthetic_run.relative_to(synthetic_root).as_posix()}),
            encoding="utf-8",
        )
        resolved_run, _resolved_result = resolve_strategy_compare_profile_run_selector(
            synthetic_root,
            {"profile_id": "selection_pit", "run": "latest"},
            audit_id="synthetic-orderable-alignment",
            required_arm_ids=("C25", "C35"),
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "profile_run_resolver_uses_current_strategy_compare_output_namespace_and_latest_manifest",
        True,
        resolved_run.name == "synthetic_orderable_alignment",
    )

    orderable = pd.DataFrame([
        {"ticker": "A", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "model_score": 0.9, "common_target_raw_r": 3.0, "score_available": True, "target_available": True},
        {"ticker": "B", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "model_score": 0.8, "common_target_raw_r": 2.0, "score_available": True, "target_available": True},
        {"ticker": "C", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "model_score": 0.7, "common_target_raw_r": 0.0, "score_available": True, "target_available": True},
        {"ticker": "A", "trade_date": "2024-01-04", "signal_date": "2024-01-03", "model_score": 0.9, "common_target_raw_r": 0.0, "score_available": True, "target_available": True},
        {"ticker": "B", "trade_date": "2024-01-04", "signal_date": "2024-01-03", "model_score": 0.8, "common_target_raw_r": 3.0, "score_available": True, "target_available": True},
        {"ticker": "C", "trade_date": "2024-01-04", "signal_date": "2024-01-03", "model_score": 0.7, "common_target_raw_r": 2.0, "score_available": True, "target_available": True},
    ])
    with tempfile.TemporaryDirectory() as tmp:
        cap_path = Path(tmp) / "score_ranking_daily_capacity.csv"
        pd.DataFrame([
            {
                "Date": "2024-01-03",
                "Resource_Aware_Direct_Score_Order_Feasible": "TRUE",
                "Resource_Aware_Max_DL_Repair_Steps": 0,
                "Resource_Aware_Max_DL_Feasible_Ascent_Steps": 0,
                "Resource_Aware_Pre_Market_Order_Limit": 2,
                "Resource_Aware_Baseline_Selected": 2,
            },
            {
                "Date": "2024-01-04",
                "Resource_Aware_Direct_Score_Order_Feasible": "False",
                "Resource_Aware_Max_DL_Repair_Steps": 1,
                "Resource_Aware_Max_DL_Feasible_Ascent_Steps": 1,
                "Resource_Aware_Pre_Market_Order_Limit": 2,
                "Resource_Aware_Baseline_Selected": 2,
            },
        ]).to_csv(cap_path, index=False)
        capacity = _capacity_frame(cap_path)

    ranking = _daily_ranking_rows(orderable, capacity)
    d1 = ranking.loc[ranking["trade_date"] == "2024-01-03"].iloc[0]
    d2 = ranking.loc[ranking["trade_date"] == "2024-01-04"].iloc[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capacity_boolean_and_stage_contract_parses_false_string_without_truthiness_bug",
        True,
        bool(
            bool(d1["direct_feasible"])
            and not bool(d2["direct_feasible"])
            and bool(d2["repair"])
            and bool(d2["ascent"])
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "orderable_ranking_uses_actual_daily_order_limit_as_dynamic_n_not_portfolio_magic_k",
        True,
        bool(
            int(d1["k"]) == 2
            and int(d2["k"]) == 2
            and math.isclose(float(d1["daily_rho"]), 1.0, abs_tol=1e-12)
            and math.isclose(float(d1["pair_accuracy"]), 1.0, abs_tol=1e-12)
            and math.isclose(float(d1["top_target_lift_r"]), 5.0 / 6.0, abs_tol=1e-12)
            and float(d2["top_target_lift_r"]) < 0.0
        ),
    )

    execution = pd.DataFrame([
        {"ticker": "A", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "chosen_qty": 100},
        {"ticker": "B", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "chosen_qty": 100},
        {"ticker": "B", "trade_date": "2024-01-04", "signal_date": "2024-01-03", "chosen_qty": 100},
        {"ticker": "C", "trade_date": "2024-01-04", "signal_date": "2024-01-03", "chosen_qty": 100},
    ])
    translation = _execution_translation(orderable, execution, capacity)
    t2 = translation.loc[translation["trade_date"] == "2024-01-04"].iloc[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "resource_translation_compares_raw_score_top_n_with_actual_chosen_actions_post_replay",
        True,
        bool(
            int(t2["action_count"]) == 2
            and math.isclose(float(t2["raw_action_overlap"]), 0.5, abs_tol=1e-12)
            and math.isclose(float(t2["action_minus_raw_target_r"]), 1.0, abs_tol=1e-12)
            and float(t2["action_minus_raw_score"]) < 0.0
            and bool(t2["repair"])
            and bool(t2["ascent"])
        ),
    )

    fake_bundle = SimpleNamespace(
        group_table=pd.DataFrame([
            {"ticker": "A", "date": "2024-01-03", "group_index": 0},
            {"ticker": "A", "date": "2024-01-04", "group_index": 1},
        ]),
        raw_target=np.array([1.5, np.nan], dtype=np.float64),
        target_valid=np.array([True, False], dtype=bool),
    )
    with patch.object(alignment_common, "load_profile_continuous_ranker_data", return_value=fake_bundle):
        target_lookup, target_calendar = _load_common_daily_target(
            root=Path("."),
            filter_id="breakout_quality_v1",
            architecture="inception_time_v1",
            profile="daily_universal_no_time_full_list_ndcg_pairwise",
        )
    mapped = _information_date_map(pd.Series(["2024-01-05"]), target_calendar)
    latest_row = target_lookup.loc[("A", "2024-01-04"), "common_target_raw_r"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "target_incomplete_latest_trading_day_is_not_silently_backfilled_to_older_valid_target",
        True,
        bool(
            list(target_calendar) == ["2024-01-03", "2024-01-04"]
            and mapped.iloc[0] == "2024-01-04"
            and pd.isna(latest_row)
        ),
    )

    summary["workflow"] = "mr13e_orderable_feasible_alignment_audit"
    summary["fixed_k_used"] = False
    return results, summary


def validate_breakout_quality_selector_stage_translation_audit_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SELECTOR_STAGE_TRANSLATION_AUDIT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config.audit import get_audit_definitions
    from tools.audit.catalog import get_audit_entry
    from tools.audit.breakout_quality.selector_stage_translation import (
        _directional_conclusion,
        _execution_stage_rows,
        _stage_daily_summary,
        _transition_rows,
        collect_selector_stage_translation_status,
    )

    definitions = {item.audit_id: item for item in get_audit_definitions("breakout_quality")}
    definition = definitions["mr13e-selector-stage-translation"]
    previous = definitions["mr13e-orderable-feasible-alignment"]
    next_definition = definitions["mr13e-minimum-repair-mechanism"]
    entry = get_audit_entry("selector_stage_translation")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selector_stage_audit_is_config_driven_read_only_and_is_now_completed_before_minimum_repair_mechanism_step",
        True,
        bool(
            not definition.enabled
            and not previous.enabled
            and next_definition.enabled
            and definition.audit_type == "selector_stage_translation"
            and definition.dimensions.get("minimum_repair_seed") is True
            and definition.dimensions.get("feasible_ascent_final") is True
            and definition.outcomes.get("stage_target_delta_r") is True
            and entry.formal
            and entry.read_only
            and entry.module == "tools.audit.breakout_quality.selector_stage_translation"
        ),
    )
    with tempfile.TemporaryDirectory() as tmp:
        blocked = collect_selector_stage_translation_status(definition, project_root=Path(tmp))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selector_stage_audit_blocks_cleanly_without_completed_trace_sidecars",
        True,
        blocked.get("status") == "BLOCKED",
    )

    stage_rows = pd.DataFrame([
        {"trade_date": "2024-01-03", "stage": "raw_top_n", "ticker": "A", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 3.0, "model_score": 0.9},
        {"trade_date": "2024-01-03", "stage": "raw_top_n", "ticker": "B", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 1.0, "model_score": 0.8},
        {"trade_date": "2024-01-03", "stage": "minimum_repair_seed", "ticker": "A", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 3.0, "model_score": 0.9},
        {"trade_date": "2024-01-03", "stage": "minimum_repair_seed", "ticker": "C", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 0.0, "model_score": 0.6},
        {"trade_date": "2024-01-03", "stage": "feasible_ascent_final", "ticker": "A", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 3.0, "model_score": 0.9},
        {"trade_date": "2024-01-03", "stage": "feasible_ascent_final", "ticker": "D", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 2.0, "model_score": 0.7},
        {"trade_date": "2024-01-03", "stage": "entry_action", "ticker": "A", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 3.0, "model_score": 0.9},
        {"trade_date": "2024-01-03", "stage": "entry_action", "ticker": "D", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 2.0, "model_score": 0.7},
        {"trade_date": "2024-01-03", "stage": "actual_fill", "ticker": "D", "signal_date": "2024-01-02", "target_available": True, "common_target_raw_r": 2.0, "model_score": 0.7},
    ])
    daily = _stage_daily_summary(stage_rows)
    transitions = _transition_rows(daily).set_index("transition")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selector_stage_target_attribution_separates_repair_ascent_action_and_fill_without_fixed_k",
        True,
        bool(
            math.isclose(float(transitions.loc["raw_to_repair", "target_delta_r"]), -0.5, abs_tol=1e-12)
            and math.isclose(float(transitions.loc["repair_to_ascent", "target_delta_r"]), 1.0, abs_tol=1e-12)
            and math.isclose(float(transitions.loc["ascent_to_action", "target_delta_r"]), 0.0, abs_tol=1e-12)
            and math.isclose(float(transitions.loc["action_to_fill", "target_delta_r"]), -0.5, abs_tol=1e-12)
            and math.isclose(float(transitions.loc["raw_to_repair", "membership_overlap"]), 0.5, abs_tol=1e-12)
        ),
    )

    execution_orderable = pd.DataFrame([
        {"ticker": "A", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "model_score": 0.9, "common_target_raw_r": 3.0, "score_available": True, "target_available": True},
        {"ticker": "B", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "model_score": 0.8, "common_target_raw_r": 1.0, "score_available": True, "target_available": True},
    ])
    execution_sidecar = pd.DataFrame([
        {"ticker": "A", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "chosen_qty": 100, "filled_qty": 100, "entry_filled": "True"},
        {"ticker": "B", "trade_date": "2024-01-03", "signal_date": "2024-01-02", "chosen_qty": 100, "filled_qty": 0, "entry_filled": "False"},
    ])
    execution_stages = _execution_stage_rows(execution_orderable, execution_sidecar)
    fill_tickers = set(execution_stages.loc[execution_stages["stage"] == "actual_fill", "ticker"].astype(str))
    action_tickers = set(execution_stages.loc[execution_stages["stage"] == "entry_action", "ticker"].astype(str))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "actual_fill_stage_uses_positive_filled_qty_and_does_not_treat_false_csv_string_as_true",
        True,
        action_tickers == {"A", "B"} and fill_tickers == {"A"},
    )

    def arm(trans):
        return {
            "repair_days_only": {
                "transitions": {
                    key: {"target_delta_r": value}
                    for key, value in trans.items()
                }
            }
        }
    payload = {
        "phases": {
            "forward_oos": {
                "arms": {
                    "C20": arm({"raw_to_repair": -0.2, "repair_to_ascent": 0.3, "ascent_to_action": -0.1, "action_to_fill": -0.1}),
                    "C29": arm({"raw_to_repair": -0.3, "repair_to_ascent": 0.1, "ascent_to_action": -0.4, "action_to_fill": -0.2}),
                    "C36": arm({"raw_to_repair": -0.25, "repair_to_ascent": -0.2, "ascent_to_action": -0.15, "action_to_fill": -0.1}),
                }
            }
        }
    }
    conclusion = _directional_conclusion(payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selector_stage_classification_uses_most_negative_forward_repair_stage_delta_without_weighted_score_or_threshold",
        True,
        bool(
            conclusion.get("classification") == "DOMINANT_EXTRA_LOSS_AT_REPAIR_TO_ASCENT"
            and conclusion.get("dominant_forward_extra_loss_transition") == "repair_to_ascent"
            and math.isclose(
                float(conclusion["mr13e_minus_mr12b_forward_repair_transition_target_delta_r"]["repair_to_ascent"]),
                -0.5,
                abs_tol=1e-12,
            )
        ),
    )

    summary["workflow"] = "mr13e_selector_stage_translation_audit"
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
    previous = definitions["mr13e-selector-stage-translation"]
    entry = get_audit_entry("minimum_repair_mechanism")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "minimum_repair_mechanism_audit_is_active_config_driven_read_only_scalable_search_certificate_step",
        True,
        bool(
            definition.enabled
            and not previous.enabled
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
    from tools.audit.breakout_quality.pass_persistence import run_pass_persistence_audit
    from tools.audit.breakout_quality.pass_quality import run_pass_quality_audit
    from tools.audit.breakout_quality.selection_confidence import run_selection_confidence_audit
    from tools.audit.breakout_quality.c15_strategy_attribution import (
        collect_strategy_attribution_status,
        run_strategy_attribution_audit,
    )
    from tools.audit.breakout_quality.strategy_realization_capture import (
        collect_strategy_realization_capture_status,
        run_strategy_realization_capture_audit,
    )
    from tools.audit.breakout_quality.forward_robustness_portfolio_translation import (
        collect_forward_robustness_portfolio_translation_status,
        run_forward_robustness_portfolio_translation_audit,
    )
    from tools.audit.breakout_quality.pit_fold_runtime_attribution import (
        build_pit_fold_runtime_attribution,
    )
    from tools.audit.breakout_quality.pit_target_realization_attribution import (
        build_pit_target_realization_attribution,
    )
    from tools.audit.catalog import validate_audit_catalog

    case_id = "AUDIT_FRAMEWORK"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    validate_audit_config()
    all_definitions = get_audit_definitions("breakout_quality")
    enabled_definitions = get_enabled_audit_definitions("breakout_quality")
    quality_definition = next(
        (item for item in all_definitions if item.audit_type == "pass_quality"), None
    )
    persistence_definition = next(
        (item for item in all_definitions if item.audit_type == "pass_persistence"), None
    )
    confidence_definition = next(
        (item for item in all_definitions if item.audit_type == "selection_confidence"), None
    )
    strategy_attribution_definition = next(
        (item for item in all_definitions if item.audit_id == "c15-strategy-attribution"), None
    )
    source_attribution_definition = next(
        (item for item in all_definitions if item.audit_id == "c15-source-attribution"), None
    )
    pit_realization_definition = next(
        (item for item in all_definitions if item.audit_id == "c23-c25-pit-realization"), None
    )
    pit_fold_runtime_definition = next(
        (item for item in all_definitions if item.audit_id == "c23-c25-pit-fold-runtime"), None
    )
    pit_target_realization_definition = next(
        (item for item in all_definitions if item.audit_id == "c23-c25-pit-target-realization"), None
    )
    portfolio_translation_definition = next(
        (item for item in all_definitions if item.audit_id == "c23-c26-pit-portfolio-translation"), None
    )
    forward_robustness_translation_definition = next(
        (item for item in all_definitions if item.audit_id == "forward-robustness-portfolio-translation"), None
    )
    cross_period_definition = next(
        (item for item in all_definitions if item.audit_id == "cross-period-year-regime-attribution"), None
    )
    orderable_alignment_definition = next(
        (item for item in all_definitions if item.audit_id == "mr13e-orderable-feasible-alignment"), None
    )
    selector_stage_definition = next(
        (item for item in all_definitions if item.audit_id == "mr13e-selector-stage-translation"), None
    )
    minimum_repair_definition = next(
        (item for item in all_definitions if item.audit_id == "mr13e-minimum-repair-mechanism"), None
    )
    validate_audit_catalog(all_definitions)
    project_root = Path(__file__).resolve().parents[2]
    audit_app_source = (project_root / "apps" / "research.py").read_text(encoding="utf-8")
    model_app_source = (project_root / "tools" / "filters" / "breakout_quality" / "application.py").read_text(encoding="utf-8")
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

    from tools.audit import primitives as audit_primitives
    from tools.audit.breakout_quality import artifact_primitives, pit_primitives
    from tools.audit.breakout_quality import selection_replay_primitives, target_statistics
    from tools.audit.breakout_quality import continuous_target as continuous_target_audit
    from tools.audit.breakout_quality import no_time_continuous_target as no_time_target_audit
    from tools.audit.breakout_quality import selection_strategy_realization as selection_realization_audit
    from tools.audit.breakout_quality import candidate_counterfactual_execution as counterfactual_audit
    from tools.audit.breakout_quality import target_component_attribution as target_component_audit
    from tools.audit.breakout_quality import pit_fold_runtime_attribution as pit_fold_audit
    from tools.audit.breakout_quality import c15_strategy_attribution as strategy_attribution_audit
    from tools.audit.breakout_quality import forward_robustness_portfolio_translation as forward_robustness_translation_audit
    from tools.audit.breakout_quality import cross_period_year_regime_attribution as cross_period_audit
    from tools.audit.sources import multi_seed_robustness as multi_seed_audit_source

    synthetic_execution_trade = pd.DataFrame([{
        "category": "candidate_only",
        "match_key": "AAA|2024-01-05|normal|1",
        "candidate_r": 2.0,
        "comparator_r": 0.0,
    }])
    synthetic_execution_rows = pd.DataFrame([{
        "execution_order": 7,
        "ticker": "AAA",
        "trade_date": "2024-01-05",
        "candidate_date": "2024-01-04",
        "signal_date": "2024-01-03",
        "entry_type": "normal",
        "entry_filled": True,
        "candidate_risk_utilization": 0.95,
        "chosen_risk_utilization": 0.80,
        "actual_risk_utilization": 0.75,
        "risk_cap_binding": True,
        "position_cap_binding": False,
        "risk_position_tie": False,
        "capital_binding": False,
        "max_qty_binding": False,
        "lot_rounding_binding": False,
        "entry_budget_cash_binding": True,
        "candidate_qty": 1000,
        "chosen_qty": 800,
        "filled_qty": 800,
        "binding_signature": "RISK_CAP+ENTRY_BUDGET_CASH",
    }])
    synthetic_execution_summary = forward_robustness_translation_audit._exclusive_execution_side_summary(
        synthetic_execution_trade,
        synthetic_execution_rows,
        category="candidate_only",
        r_col="candidate_r",
        side_label="candidate",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_robustness_execution_binding_uses_canonical_trade_identity_and_preserves_binding_counts",
        True,
        synthetic_execution_summary["trade_count"] == 1
        and math.isclose(float(synthetic_execution_summary["actual_risk_utilization_mean"]), 0.75)
        and synthetic_execution_summary["cash_binding_count"] == 1
        and synthetic_execution_summary["risk_cap_binding_count"] == 1
        and synthetic_execution_summary["binding_signature_counts"] == {"RISK_CAP+ENTRY_BUDGET_CASH": 1},
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "continuous_target_audits_share_one_statistics_implementation",
        (True, True, True),
        (
            continuous_target_audit._distribution_metrics is target_statistics.distribution_metrics,
            no_time_target_audit._distribution_metrics is target_statistics.distribution_metrics,
            continuous_target_audit._daily_rankability is no_time_target_audit._daily_rankability,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "selection_and_pit_audits_share_identity_and_artifact_primitives",
        (True, True, True, True),
        (
            selection_realization_audit._attach_targets is selection_replay_primitives.attach_targets,
            counterfactual_audit._attach_targets is selection_replay_primitives.attach_targets,
            pit_fold_audit._candidate_pit_identity is pit_primitives.candidate_pit_identity,
            target_component_audit._sha256_file is audit_primitives.sha256_file
            and artifact_primitives.sha256_file is audit_primitives.sha256_file,
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_robustness_translation_reuses_public_canonical_strategy_attribution_primitive",
        True,
        forward_robustness_translation_audit.build_strategy_attribution_pair_payload
        is strategy_attribution_audit.build_strategy_attribution_pair_payload,
    )

    synthetic_cross_seed = {
        "selection_pit": pd.DataFrame([
            {"seed_order": 1, "seed": 101, "delta_direct_selection_r": -3.0, "delta_romd": -1.0, "direct_pair_exclusive_delta_r": -2.0, "selection_basis_gap_r": 1.0},
            {"seed_order": 2, "seed": 202, "delta_direct_selection_r": -1.0, "delta_romd": -0.5, "direct_pair_exclusive_delta_r": -1.0, "selection_basis_gap_r": 0.0},
        ]),
        "forward_oos": pd.DataFrame([
            {"seed_order": 1, "seed": 101, "delta_direct_selection_r": 4.0, "delta_romd": 2.0, "direct_pair_exclusive_delta_r": 3.0, "selection_basis_gap_r": -1.0},
            {"seed_order": 2, "seed": 202, "delta_direct_selection_r": 2.0, "delta_romd": 1.0, "direct_pair_exclusive_delta_r": 2.0, "selection_basis_gap_r": 0.0},
        ]),
    }
    synthetic_cross_yearly = pd.DataFrame([
        {"phase_id": "selection_pit", "year": 2017, "direct_pair_exclusive_delta_r_mean": -5.0},
        {"phase_id": "selection_pit", "year": 2018, "direct_pair_exclusive_delta_r_mean": 1.0},
        {"phase_id": "forward_oos", "year": 2023, "direct_pair_exclusive_delta_r_mean": 4.0},
        {"phase_id": "forward_oos", "year": 2024, "direct_pair_exclusive_delta_r_mean": 2.0},
    ])
    synthetic_cross_phase_aggregate = {
        phase_id: cross_period_audit._phase_aggregate(
            frame,
            synthetic_cross_yearly[synthetic_cross_yearly["phase_id"] == phase_id],
        )
        for phase_id, frame in synthetic_cross_seed.items()
    }
    synthetic_cross_interpretation = cross_period_audit._interpretation(
        synthetic_cross_phase_aggregate, synthetic_cross_yearly
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "cross_period_audit_is_config_driven_read_only_and_shares_one_multi_seed_source_resolver",
        True,
        bool(
            cross_period_definition is not None
            and cross_period_definition.audit_type == "robustness_cross_period_attribution"
            and not cross_period_definition.enabled
            and dict(cross_period_definition.source).get("kind") == "multi_seed_robustness_cross_period"
            and forward_robustness_translation_audit.resolve_two_arm_multi_seed_source
                is multi_seed_audit_source.resolve_two_arm_multi_seed_source
            and cross_period_audit.resolve_two_arm_multi_seed_source
                is multi_seed_audit_source.resolve_two_arm_multi_seed_source
            and forward_robustness_translation_audit.compact_artifacts_from_unit
                is multi_seed_audit_source.compact_artifacts_from_unit
            and cross_period_audit.compact_artifacts_from_unit
                is multi_seed_audit_source.compact_artifacts_from_unit
            and synthetic_cross_interpretation["classification"]
                == "RANKING_EDGE_DIRECTION_REVERSAL"
            and int(synthetic_cross_interpretation["selection_worst_year"]["year"]) == 2017
            and synthetic_cross_interpretation["yearly_basis_note"].startswith("逐年度只使用")
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "project_audit_app_and_breakout_quality_facade_share_catalog_runner_and_cli_smoke_registry",
        True,
        "from tools.audit.runner import" in audit_app_source
        and "get_active_audit_module_id" in audit_app_source
        and "get_domain_cli_commands" in model_app_source
        and "_interactive_audit_menu" not in model_app_source
        and '([sys.executable, "apps/research.py", "audit", "--help"]' in quick_gate_source
        and '"apps/research.py",' in quick_gate_source.split("INLINE_CLI_TARGETS = {", 1)[1],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "audit_config_and_catalog_support_project_modules_and_config_driven_handlers",
        True,
        quality_definition is not None
        and persistence_definition is not None
        and confidence_definition is not None
        and strategy_attribution_definition is not None
        and source_attribution_definition is not None
        and pit_realization_definition is not None
        and pit_fold_runtime_definition is not None
        and pit_target_realization_definition is not None
        and portfolio_translation_definition is not None
        and portfolio_translation_definition.outcomes.get("risk_dollar_translation") is True
        and forward_robustness_translation_definition is not None
        and forward_robustness_translation_definition.audit_type == "robustness_portfolio_translation"
        and forward_robustness_translation_definition.source.get("kind") == "multi_seed_robustness"
        and forward_robustness_translation_definition.outcomes.get("all_seed_required") is True
        and isinstance(forward_robustness_translation_definition.enabled, bool)
        and isinstance(portfolio_translation_definition.enabled, bool)
        and orderable_alignment_definition is not None
        and not orderable_alignment_definition.enabled
        and orderable_alignment_definition.audit_type == "orderable_feasible_alignment"
        and orderable_alignment_definition.source.get("kind") == "strategy_compare_cross_phase"
        and orderable_alignment_definition.outcomes.get("no_fixed_k") is True
        and selector_stage_definition is not None
        and not selector_stage_definition.enabled
        and selector_stage_definition.audit_type == "selector_stage_translation"
        and selector_stage_definition.source.get("kind") == "strategy_compare_cross_phase"
        and minimum_repair_definition is not None
        and minimum_repair_definition.enabled
        and minimum_repair_definition.audit_type == "minimum_repair_mechanism"
        and minimum_repair_definition.source.get("kind") == "strategy_compare_cross_phase"
        and minimum_repair_definition.outcomes.get("frozen_score_only_search_certificate") is True
        and minimum_repair_definition.outcomes.get("no_numeric_threshold") is True
        and "breakout_quality" in get_audit_module_ids(enabled_only=True)
        and bool(enabled_definitions)
        and all(bool(str(item.source.get("kind") or "").strip()) for item in all_definitions),
    )

    if forward_robustness_translation_definition is None:
        forward_status_blocked = False
    else:
        with tempfile.TemporaryDirectory() as tmp:
            forward_status = collect_forward_robustness_portfolio_translation_status(
                forward_robustness_translation_definition,
                project_root=Path(tmp),
            )
        forward_status_blocked = (
            forward_status.get("status") == "BLOCKED"
            and "candidate_arm_id" in dict(forward_status.get("source") or {})
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_robustness_portfolio_translation_audit_is_read_only_and_blocks_when_compact_source_is_missing",
        True,
        forward_status_blocked,
    )

    forward_e2e_ok = False
    if forward_robustness_translation_definition is not None:
        from config.strategy_compare import (
            get_strategy_comparison_settings as get_compare_settings,
            get_strategy_multi_seed_robustness_settings as get_robustness_settings,
        )
        from filters.breakout_quality import strategy_multi_seed_robustness as robustness_source_module

        source_cfg = dict(forward_robustness_translation_definition.source)
        robustness_id = str(source_cfg["robustness_id"])
        robustness_cfg = get_robustness_settings(robustness_id)
        compare_settings = get_compare_settings(robustness_cfg.profile_id)
        candidate_arm_id = str(source_cfg["candidate_arm_id"])
        comparator_arm_id = str(source_cfg["comparator_arm_id"])
        synthetic_seed_values = (101, 202)
        with tempfile.TemporaryDirectory() as tmp:
            synthetic_root = Path(tmp).resolve()
            synthetic_run_root = synthetic_root / robustness_cfg.output_root / "syntheticfp"
            synthetic_run_root.mkdir(parents=True, exist_ok=True)
            result_rows = []
            for arm_order, arm_id in enumerate((comparator_arm_id, candidate_arm_id), start=1):
                arm = compare_settings.arms[arm_id]
                dl = compare_settings.dl_sources[str(arm.dl_id)]
                runtime_spec = robustness_source_module._arm_runtime_spec(arm)
                active_prefix = str(runtime_spec["active_key"])
                for seed_order, seed in enumerate(synthetic_seed_values, start=1):
                    pair_dir = synthetic_root / "pairs" / f"{arm_id}_{seed}"
                    pair_dir.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(columns=["Date", "Ticker", "Type"]).to_csv(
                        pair_dir / f"{active_prefix}_trades.csv", index=False, encoding="utf-8-sig"
                    )
                    pd.DataFrame([
                        {"Date": "2024-01-02", "Equity": 100000.0},
                        {"Date": "2024-01-03", "Equity": 100000.0},
                    ]).to_csv(
                        pair_dir / f"{active_prefix}_equity.csv", index=False, encoding="utf-8-sig"
                    )
                    pd.DataFrame([
                        {
                            "Date": "2024-01-02",
                            "Post_Execution_Positions": 0,
                            "End_Position_Gap": 10,
                            "Filled_Buys_Today": 0,
                            "Missed_Buys_Today": 0,
                        },
                        {
                            "Date": "2024-01-03",
                            "Post_Execution_Positions": 0,
                            "End_Position_Gap": 10,
                            "Filled_Buys_Today": 0,
                            "Missed_Buys_Today": 0,
                        },
                    ]).to_csv(
                        pair_dir / f"{active_prefix}_daily_capacity.csv", index=False, encoding="utf-8-sig"
                    )
                    pd.DataFrame(columns=["ticker", "trade_date", "signal_date"]).to_csv(
                        pair_dir / f"{active_prefix}_selected_buys.csv", index=False, encoding="utf-8-sig"
                    )
                    pd.DataFrame(columns=[
                        "execution_order", "ticker", "trade_date", "candidate_date", "signal_date", "entry_type",
                        "candidate_qty", "chosen_qty", "filled_qty", "entry_filled",
                        "candidate_risk_utilization", "chosen_risk_utilization", "actual_risk_utilization",
                        "risk_cap_binding", "position_cap_binding", "risk_position_tie",
                        "capital_binding", "max_qty_binding", "lot_rounding_binding",
                        "entry_budget_cash_binding", "binding_signature",
                    ]).to_csv(
                        pair_dir / f"{active_prefix}_execution.csv", index=False, encoding="utf-8-sig"
                    )
                    attribution_job = {
                        "scientific_fingerprint": "syntheticfp",
                        "robustness_id": robustness_id,
                        "profile_id": robustness_cfg.profile_id,
                        "arm_order": arm_order,
                        "seed": seed,
                        "seed_order": seed_order,
                        "comparison_start": "2024-01-01",
                        "comparison_end": "2024-12-31",
                        "selected_epoch": 1,
                        "fold_count": None,
                        "model_sha256": "synthetic-model",
                        "score_sha256": "synthetic-score",
                    }
                    with patch.object(robustness_source_module, "PROJECT_ROOT", synthetic_root):
                        robustness_source_module._write_compact_attribution_source(
                            pair_dir=pair_dir,
                            destination_dir=robustness_source_module._attribution_unit_dir(
                                synthetic_run_root, arm_id, seed
                            ),
                            job=attribution_job,
                            arm=arm,
                            dl=dl,
                            runtime_spec=runtime_spec,
                        )
                        robustness_source_module._mark_attribution_unit_verified(
                            synthetic_run_root,
                            arm_id=arm_id,
                            seed=seed,
                            expected_fingerprint="syntheticfp",
                            source="synthetic_formal_contract",
                        )
                    result_rows.append({
                        "arm_id": arm_id,
                        "name": arm.name,
                        "seed": seed,
                        "seed_order": seed_order,
                        "arm_order": arm_order,
                        "total_return_pct": 0.0,
                        "max_drawdown_pct": 1.0,
                        "return_over_max_drawdown": 0.0,
                        "expected_value_r": 0.0,
                        # Same-param direct-selection R is baseline-relative per arm.
                        # Deliberately make candidate-comparator delta non-zero while
                        # direct pair trade partition remains empty (=0) to prove the
                        # Audit does not conflate the two attribution bases.
                        "direct_selection_r": 5.0 if arm_id == candidate_arm_id else 1.0,
                    })
            pd.DataFrame(result_rows).to_csv(
                synthetic_run_root / robustness_source_module.SEED_RESULTS_FILENAME,
                index=False,
                encoding="utf-8-sig",
            )
            synthetic_contract = {
                "fingerprint": "syntheticfp",
                "robustness_id": robustness_id,
                "profile_id": robustness_cfg.profile_id,
                "seed_count": len(synthetic_seed_values),
                "resolved_seeds": list(synthetic_seed_values),
            }
            (synthetic_run_root / robustness_source_module.MANIFEST_FILENAME).write_text(
                json.dumps({"status": "COMPLETED", "contract": synthetic_contract}),
                encoding="utf-8",
            )
            (synthetic_run_root / robustness_source_module.SUMMARY_FILENAME).write_text(
                json.dumps({"fingerprint": "syntheticfp", "contract": synthetic_contract}),
                encoding="utf-8",
            )
            latest_path = synthetic_root / robustness_cfg.output_root / robustness_source_module.LATEST_FILENAME
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(
                json.dumps({
                    "summary_path": (synthetic_run_root / robustness_source_module.SUMMARY_FILENAME)
                    .relative_to(synthetic_root)
                    .as_posix()
                }),
                encoding="utf-8",
            )
            synthetic_status = collect_forward_robustness_portfolio_translation_status(
                forward_robustness_translation_definition,
                project_root=synthetic_root,
            )
            synthetic_payload = run_forward_robustness_portfolio_translation_audit(
                forward_robustness_translation_definition,
                project_root=synthetic_root,
                quiet=True,
            )
            latest_audit = (
                synthetic_root
                / AUDIT_OUTPUT_ROOT
                / forward_robustness_translation_definition.output_subdir
                / "latest"
            )
            synthetic_seed_summary = list(synthetic_payload.get("seed_summary") or [])
            forward_e2e_ok = (
                synthetic_status.get("status") == "READY"
                and int(synthetic_payload.get("schema_version") or 0) == 5
                and int(synthetic_payload["metadata"]["seed_count"]) == len(synthetic_seed_values)
                and synthetic_payload["metadata"]["training_performed"] is False
                and synthetic_payload["metadata"]["portfolio_replay_executed"] is False
                and len(synthetic_seed_summary) == len(synthetic_seed_values)
                and all(math.isclose(float(row["delta_direct_selection_r"]), 4.0, abs_tol=1e-12) for row in synthetic_seed_summary)
                and all(math.isclose(float(row["direct_pair_exclusive_delta_r"]), 0.0, abs_tol=1e-12) for row in synthetic_seed_summary)
                and all(math.isclose(float(row["selection_basis_gap_r"]), -4.0, abs_tol=1e-12) for row in synthetic_seed_summary)
                and all("candidate_only_avg_implied_initial_risk" in row for row in synthetic_seed_summary)
                and all("comparator_only_avg_implied_initial_risk" in row for row in synthetic_seed_summary)
                and all("exclusive_risk_weighted_r_gap" in row for row in synthetic_seed_summary)
                and all("exclusive_equal_risk_selection_effect_pnl" in row for row in synthetic_seed_summary)
                and all("exclusive_average_risk_scale_effect_pnl" in row for row in synthetic_seed_summary)
                and all("exclusive_within_set_weighting_effect_pnl" in row for row in synthetic_seed_summary)
                and all(abs(float(row["exclusive_bridge_residual_pnl"])) <= 1e-8 for row in synthetic_seed_summary)
                and "Exclusive trade R→Dollar bridge" in (latest_audit / "audit.md").read_text(encoding="utf-8")
                and "Exclusive ΔPnL exact decomposition" in (latest_audit / "audit.md").read_text(encoding="utf-8")
                and "Exclusive risk utilization / binding attribution" in (latest_audit / "audit.md").read_text(encoding="utf-8")
                and (latest_audit / "execution_binding.csv.gz").is_file()
                and (latest_audit / "audit.json").is_file()
                and (latest_audit / "seed_summary.csv").is_file()
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_robustness_portfolio_translation_runs_all_resolved_seeds_from_verified_compact_source_without_training_or_replay",
        True,
        forward_e2e_ok,
    )

    synthetic_scores = pd.DataFrame([
        {"ticker": "A", "date": "2023-12-20", "fold_id": "F1", "breakout_quality_score": 0.70},
        {"ticker": "B", "date": "2023-12-21", "fold_id": "F1", "breakout_quality_score": 0.60},
        {"ticker": "C", "date": "2024-01-02", "fold_id": "F2", "breakout_quality_score": 0.80},
        {"ticker": "D", "date": "2024-01-03", "fold_id": "F2", "breakout_quality_score": 0.50},
    ])
    synthetic_orderable = pd.DataFrame([
        {"ticker": "A", "trade_date": "2024-01-04", "signal_date": "2023-12-20", "breakout_quality_score_date": "2023-12-20"},
        {"ticker": "C", "trade_date": "2024-01-04", "signal_date": "2024-01-02", "breakout_quality_score_date": "2024-01-02"},
        {"ticker": "C", "trade_date": "2024-03-15", "signal_date": "2024-01-02", "breakout_quality_score_date": "2024-01-02"},
        {"ticker": "D", "trade_date": "2024-03-15", "signal_date": "2024-01-03", "breakout_quality_score_date": "2024-01-03"},
    ])
    synthetic_trade_contributions = pd.DataFrame([
        {"category": "comparator_only", "ticker": "A", "entry_date": "2024-01-04", "signal_date": "2023-12-20", "candidate_r": 0.0, "comparator_r": 2.0, "r_delta": -2.0},
        {"category": "candidate_only", "ticker": "C", "entry_date": "2024-01-04", "signal_date": "2024-01-02", "candidate_r": -1.0, "comparator_r": 0.0, "r_delta": -1.0},
        {"category": "comparator_only", "ticker": "D", "entry_date": "2024-03-15", "signal_date": "2024-01-03", "candidate_r": 0.0, "comparator_r": -1.0, "r_delta": 1.0},
        {"category": "candidate_only", "ticker": "C", "entry_date": "2024-03-15", "signal_date": "2024-01-02", "candidate_r": 1.0, "comparator_r": 0.0, "r_delta": 1.0},
    ])
    fold_payload, fold_frames = build_pit_fold_runtime_attribution(
        score_table=synthetic_scores,
        pit_audit={
            "fold_drift": {"drift_flag": True, "flagged_folds": ["F2"], "max_adjacent_mean_shift_in_pooled_std": 1.2},
            "fold_metrics": [],
        },
        orderable_candidates=synthetic_orderable,
        trade_contributions=synthetic_trade_contributions,
        fold_boundary_window_days=30,
    )
    by_scope = {row["scope"]: row for row in fold_payload["exclusive_selection"]["by_day_scope"]}
    by_boundary = {str(row["scope"]): row for row in fold_payload["exclusive_selection"]["by_fold_boundary_window"]}
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pit_fold_runtime_attribution_separates_mixed_fold_and_boundary_winner_capture_without_replay",
        True,
        fold_payload["runtime_mixing"]["mixed_fold_day_count"] == 1
        and math.isclose(float(fold_payload["runtime_mixing"]["cross_fold_pair_share"]), 0.5, abs_tol=1e-12)
        and math.isclose(float(by_scope["mixed_fold"]["selection_delta_r"]), -3.0, abs_tol=1e-12)
        and math.isclose(float(by_scope["single_fold"]["selection_delta_r"]), 2.0, abs_tol=1e-12)
        and math.isclose(float(by_scope["mixed_fold"]["winner_r_contribution_delta"]), -2.0, abs_tol=1e-12)
        and math.isclose(float(by_boundary["True"]["selection_delta_r"]), -3.0, abs_tol=1e-12)
        and math.isclose(float(by_boundary["False"]["selection_delta_r"]), 2.0, abs_tol=1e-12)
        and not fold_frames["exclusive_trade_fold_attribution"].empty,
    )

    target_payload, target_frames = build_pit_target_realization_attribution(
        score_table=pd.DataFrame([
            {"ticker": "A", "date": "2024-01-01", "group_index": 0, "breakout_quality_score": 0.40},
            {"ticker": "B", "date": "2024-01-02", "group_index": 1, "breakout_quality_score": 0.90},
            {"ticker": "C", "date": "2024-01-03", "group_index": 2, "breakout_quality_score": 0.50},
            {"ticker": "D", "date": "2024-01-04", "group_index": 3, "breakout_quality_score": 0.95},
        ]),
        target_raw_r=np.asarray([0.5, 2.0, 0.6, 2.2], dtype=np.float32),
        target_valid_mask=np.asarray([True, True, True, True], dtype=bool),
        trade_contributions=pd.DataFrame([
            {"category": "comparator_only", "ticker": "A", "entry_date": "2024-01-03", "signal_date": "2024-01-01", "candidate_r": 0.0, "comparator_r": 2.0, "r_delta": -2.0},
            {"category": "candidate_only", "ticker": "B", "entry_date": "2024-01-04", "signal_date": "2024-01-02", "candidate_r": 1.0, "comparator_r": 0.0, "r_delta": 1.0},
            {"category": "comparator_only", "ticker": "C", "entry_date": "2024-03-20", "signal_date": "2024-01-03", "candidate_r": 0.0, "comparator_r": 2.0, "r_delta": -2.0},
            {"category": "candidate_only", "ticker": "D", "entry_date": "2024-03-22", "signal_date": "2024-01-04", "candidate_r": -1.0, "comparator_r": 0.0, "r_delta": -1.0},
        ]),
        score_age_quantile_groups=2,
    )
    age_rows = target_payload["age_buckets"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pit_target_realization_attribution_detects_high_target_but_worse_realized_r_and_age_concentration_without_counterfactual",
        True,
        target_payload["covered_trade_count"] == 4
        and float(target_payload["candidate_only"]["avg_target_r"]) > float(target_payload["baseline_only"]["avg_target_r"])
        and float(target_payload["candidate_only"]["avg_realized_r"]) < float(target_payload["baseline_only"]["avg_realized_r"])
        and len(age_rows) == 2
        and float(age_rows[-1]["selection_delta_r"]) < float(age_rows[0]["selection_delta_r"])
        and target_payload["semantic_boundary"]["unselected_counterfactual_r_available"] is False
        and not target_frames["exclusive_trade_target_realization"].empty,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "outputs" / "strategy_compare" / "runs" / "synthetic"
        latest_dir = root / "outputs" / "strategy_compare" / "latest"
        pair_dir = (
            run_dir
            / "pairs"
            / "min_roos__all_off__A9__resource_aware_binary_basket"
        )
        events_dir = (
            root
            / "outputs"
            / "filters"
            / "breakout_quality"
            / "breakout_quality_v1"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        events_dir.mkdir(parents=True, exist_ok=True)

        result_payload = {
            "status": "COMPLETED",
            "settings": {
                "arms": {
                    "C12": {
                        "arm_id": "C12",
                        "enabled": True,
                        "param_source": "min_roos",
                        "rule_policy": "all_off",
                        "dl_enabled": True,
                        "dl_id": "A9",
                        "dl_runtime_mode": "resource-aware-binary-basket",
                    }
                },
                "dl_sources": {
                    "A9": {
                        "dl_id": "A9",
                        "filter_id": "breakout_quality_v1",
                        "threshold": 0.5,
                    }
                },
            }
        }
        (run_dir / "strategy_comparison.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "strategy_comparison.json").write_text(
            json.dumps(result_payload), encoding="utf-8"
        )
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/synthetic"}),
            encoding="utf-8",
        )

        orderable = pd.DataFrame(
            [
                {
                    "ticker": "A",
                    "trade_date": "2026-01-02",
                    "candidate_date": "2026-01-02",
                    "signal_date": "2026-01-01",
                    "candidate_type": "normal",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "A",
                    "trade_date": "2026-01-03",
                    "candidate_date": "2026-01-03",
                    "signal_date": "2026-01-01",
                    "candidate_type": "extended",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "A",
                    "trade_date": "2026-01-04",
                    "candidate_date": "2026-01-04",
                    "signal_date": "2026-01-01",
                    "candidate_type": "extended",
                    "high_len": 200,
                    "breakout_quality_score": 0.55,
                },
                {
                    "ticker": "B",
                    "trade_date": "2026-01-05",
                    "candidate_date": "2026-01-05",
                    "signal_date": "2026-01-02",
                    "candidate_type": "extended",
                    "high_len": 205,
                    "breakout_quality_score": 0.65,
                },
                {
                    "ticker": "C",
                    "trade_date": "2026-01-08",
                    "candidate_date": "2026-01-08",
                    "signal_date": "2026-01-03",
                    "candidate_type": "extended",
                    "high_len": 210,
                    "breakout_quality_score": 0.75,
                },
                {
                    "ticker": "D",
                    "trade_date": "2026-01-11",
                    "candidate_date": "2026-01-11",
                    "signal_date": "2026-01-04",
                    "candidate_type": "normal",
                    "high_len": 215,
                    "breakout_quality_score": 0.85,
                },
                {
                    "ticker": "E",
                    "trade_date": "2026-01-12",
                    "candidate_date": "2026-01-12",
                    "signal_date": "2026-01-05",
                    "candidate_type": "normal",
                    "high_len": 220,
                    "breakout_quality_score": 0.45,
                },
            ]
        )
        orderable.to_csv(
            pair_dir / "score_ranking_orderable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "B", "trade_date": "2026-01-05", "signal_date": "2026-01-02", "type": "買進 (extended)"},
                {"ticker": "D", "trade_date": "2026-01-11", "signal_date": "2026-01-04", "type": "買進 (normal)"},
            ]
        ).to_csv(
            pair_dir / "score_ranking_selected_buys.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-01-05", "Ticker": "B", "Type": "買進 (extended)", "進場類型": "extended", "候選類型": "extended", "買訊日": "2026-01-02", "候選日": "2026-01-05", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-01-20", "Ticker": "B", "Type": "全倉結算", "成交價": 110.0, "該筆總損益": 10000.0, "R_Multiple": 1.0},
                {"Date": "2026-01-11", "Ticker": "D", "Type": "買進 (normal)", "進場類型": "normal", "候選類型": "normal", "買訊日": "2026-01-04", "候選日": "2026-01-11", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-01-25", "Ticker": "D", "Type": "全倉結算", "成交價": 120.0, "該筆總損益": 20000.0, "R_Multiple": 2.0},
            ]
        ).to_csv(
            pair_dir / "score_ranking_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "A", "date": "2026-01-01", "high_len": 200, "label": 0, "label_status": "REJECT", "label_reason": "ratio", "decision_mfe_return": 0.03, "decision_mae_return": 0.04},
                {"ticker": "B", "date": "2026-01-02", "high_len": 205, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
                {"ticker": "C", "date": "2026-01-03", "high_len": 210, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.09, "decision_mae_return": 0.02},
                {"ticker": "D", "date": "2026-01-04", "high_len": 215, "label": 1, "label_status": "PASS", "label_reason": "pass", "decision_mfe_return": 0.12, "decision_mae_return": 0.01},
                {"ticker": "E", "date": "2026-01-05", "high_len": 220, "label": 0, "label_status": "REJECT", "label_reason": "mfe", "decision_mfe_return": 0.02, "decision_mae_return": 0.03},
            ]
        ).to_csv(events_dir / "events.csv", index=False, encoding="utf-8-sig")

        if quality_definition is None or persistence_definition is None:
            raise AssertionError("synthetic audit definitions missing")
        synthetic_definition = type(quality_definition)(
            module_id=quality_definition.module_id,
            audit_id=quality_definition.audit_id,
            enabled=True,
            audit_type="pass_quality",
            description=quality_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(quality_definition.dimensions),
            outcomes=dict(quality_definition.outcomes),
            output_subdir=quality_definition.output_subdir,
        )
        payload = run_pass_quality_audit(
            synthetic_definition, project_root=root, quiet=True
        )
        latest_audit = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(synthetic_definition.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "pass_quality_audit_is_read_only_config_driven_and_outputs_expected_diagnostics",
            True,
            payload["overview"]["orderable_candidate_count"] == 7
            and payload["overview"]["pass_candidate_count"] == 6
            and payload["overview"]["selected_pass_count"] == 2
            and math.isclose(payload["overview"]["selected_pass_realized_r_mean"], 1.5)
            and payload["semantic_contract"]["strategy_owns_candidate_validity"] is True
            and payload["semantic_contract"]["dl_reject_does_not_invalidate_candidate"] is True
            and bool(payload["score_groups"])
            and bool(payload["age_groups"])
            and bool(payload["candidate_type_groups"])
            and (latest_audit / "audit.md").is_file()
            and (latest_audit / "pass_candidates.csv").is_file(),
        )

        persistence_synthetic = type(persistence_definition)(
            module_id=persistence_definition.module_id,
            audit_id=persistence_definition.audit_id,
            enabled=True,
            audit_type="pass_persistence",
            description=persistence_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(persistence_definition.dimensions),
            outcomes=dict(persistence_definition.outcomes),
            output_subdir=persistence_definition.output_subdir,
        )
        persistence_payload = run_pass_persistence_audit(
            persistence_synthetic, project_root=root, quiet=True
        )
        persistence_latest = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(persistence_synthetic.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "pass_persistence_audit_detects_false_positive_candidate_day_amplification_without_expiry_semantics",
            True,
            persistence_payload["overview"]["unique_pass_event_count"] == 4
            and persistence_payload["overview"]["pass_candidate_day_count"] == 6
            and math.isclose(
                persistence_payload["overview"]["unique_event_label_pass_rate_pct"], 75.0
            )
            and math.isclose(
                persistence_payload["overview"]["candidate_day_weighted_label_pass_rate_pct"],
                50.0,
            )
            and math.isclose(
                persistence_payload["amplification"]["false_positive_candidate_day_amplification_ratio"],
                2.0,
            )
            and persistence_payload["semantic_contract"]["persistence_does_not_define_candidate_expiry"] is True
            and bool(persistence_payload["label_persistence_groups"])
            and (persistence_latest / "audit.md").is_file()
            and (persistence_latest / "event_persistence.csv").is_file(),
        )


        if confidence_definition is None:
            raise AssertionError("selection confidence audit definition missing")
        confidence_run_dir = root / "outputs" / "strategy_compare" / "runs" / "confidence_synthetic"
        confidence_pair_dir = (
            confidence_run_dir
            / "pairs"
            / "min_roos__all_off__A9__resource_aware_binary_basket"
        )
        confidence_pair_dir.mkdir(parents=True, exist_ok=True)
        (confidence_run_dir / "strategy_comparison.json").write_text(
            json.dumps(result_payload), encoding="utf-8"
        )
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/confidence_synthetic"}),
            encoding="utf-8",
        )
        confidence_orderable = pd.DataFrame(
            [
                {"ticker": "F", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 200, "breakout_quality_score": 0.51},
                {"ticker": "G", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 205, "breakout_quality_score": 0.61},
                {"ticker": "H", "trade_date": "2026-02-02", "candidate_date": "2026-02-02", "signal_date": "2026-02-01", "candidate_type": "normal", "high_len": 210, "breakout_quality_score": 0.71},
                {"ticker": "I", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 215, "breakout_quality_score": 0.52},
                {"ticker": "J", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 220, "breakout_quality_score": 0.62},
                {"ticker": "K", "trade_date": "2026-02-03", "candidate_date": "2026-02-03", "signal_date": "2026-02-02", "candidate_type": "extended", "high_len": 225, "breakout_quality_score": 0.72},
                {"ticker": "L", "trade_date": "2026-02-04", "candidate_date": "2026-02-04", "signal_date": "2026-02-03", "candidate_type": "normal", "high_len": 230, "breakout_quality_score": 0.99},
                {"ticker": "M", "trade_date": "2026-02-04", "candidate_date": "2026-02-04", "signal_date": "2026-02-03", "candidate_type": "normal", "high_len": 235, "breakout_quality_score": 0.98},
            ]
        )
        confidence_orderable.to_csv(
            confidence_pair_dir / "score_ranking_orderable_candidates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "G", "trade_date": "2026-02-02", "signal_date": "2026-02-01", "type": "買進 (normal)"},
                {"ticker": "H", "trade_date": "2026-02-02", "signal_date": "2026-02-01", "type": "買進 (normal)"},
                {"ticker": "J", "trade_date": "2026-02-03", "signal_date": "2026-02-02", "type": "買進 (extended)"},
                {"ticker": "K", "trade_date": "2026-02-03", "signal_date": "2026-02-02", "type": "買進 (extended)"},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_selected_buys.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-02-02", "Ticker": "G", "Type": "買進 (normal)", "買訊日": "2026-02-01", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-10", "Ticker": "G", "Type": "全倉結算", "成交價": 110.0, "該筆總損益": 10000.0, "R_Multiple": 1.0},
                {"Date": "2026-02-02", "Ticker": "H", "Type": "買進 (normal)", "買訊日": "2026-02-01", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-11", "Ticker": "H", "Type": "全倉結算", "成交價": 120.0, "該筆總損益": 20000.0, "R_Multiple": 2.0},
                {"Date": "2026-02-03", "Ticker": "J", "Type": "買進 (extended)", "買訊日": "2026-02-02", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-12", "Ticker": "J", "Type": "全倉結算", "成交價": 105.0, "該筆總損益": 5000.0, "R_Multiple": 0.5},
                {"Date": "2026-02-03", "Ticker": "K", "Type": "買進 (extended)", "買訊日": "2026-02-02", "成交價": 100.0, "該筆總損益": 0.0, "R_Multiple": 0.0},
                {"Date": "2026-02-13", "Ticker": "K", "Type": "全倉結算", "成交價": 115.0, "該筆總損益": 15000.0, "R_Multiple": 1.5},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_trades.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"Date": "2026-02-02", "Resource_Aware_Mode": "dl-selection"},
                {"Date": "2026-02-03", "Resource_Aware_Mode": "dl-selection"},
                {"Date": "2026-02-04", "Resource_Aware_Mode": "capital-utilization"},
            ]
        ).to_csv(
            confidence_pair_dir / "score_ranking_daily_capacity.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(
            [
                {"ticker": "F", "date": "2026-02-01", "high_len": 200, "label": 0, "decision_mfe_return": 0.02, "decision_mae_return": 0.04},
                {"ticker": "G", "date": "2026-02-01", "high_len": 205, "label": 1, "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
                {"ticker": "H", "date": "2026-02-01", "high_len": 210, "label": 1, "decision_mfe_return": 0.10, "decision_mae_return": 0.01},
                {"ticker": "I", "date": "2026-02-02", "high_len": 215, "label": 0, "decision_mfe_return": 0.02, "decision_mae_return": 0.05},
                {"ticker": "J", "date": "2026-02-02", "high_len": 220, "label": 1, "decision_mfe_return": 0.07, "decision_mae_return": 0.02},
                {"ticker": "K", "date": "2026-02-02", "high_len": 225, "label": 1, "decision_mfe_return": 0.09, "decision_mae_return": 0.01},
                {"ticker": "L", "date": "2026-02-03", "high_len": 230, "label": 0, "decision_mfe_return": 0.01, "decision_mae_return": 0.05},
                {"ticker": "M", "date": "2026-02-03", "high_len": 235, "label": 1, "decision_mfe_return": 0.08, "decision_mae_return": 0.02},
            ]
        ).to_csv(events_dir / "events.csv", index=False, encoding="utf-8-sig")

        confidence_synthetic = type(confidence_definition)(
            module_id=confidence_definition.module_id,
            audit_id=confidence_definition.audit_id,
            enabled=True,
            audit_type="selection_confidence",
            description=confidence_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "arm_id": "C12"},
            dimensions=dict(confidence_definition.dimensions),
            outcomes=dict(confidence_definition.outcomes),
            output_subdir=confidence_definition.output_subdir,
        )
        confidence_payload = run_selection_confidence_audit(
            confidence_synthetic, project_root=root, quiet=True
        )
        confidence_latest = (
            root
            / Path(AUDIT_OUTPUT_ROOT)
            / Path(confidence_synthetic.output_subdir)
            / "latest"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_confidence_audit_scopes_to_dl_selection_competition_and_measures_within_day_label_and_realized_r_ordering",
            True,
            confidence_payload["overview"]["dl_selection_day_count"] == 2
            and confidence_payload["overview"]["competition_day_count"] == 2
            and confidence_payload["overview"]["competition_pass_candidate_count"] == 6
            and confidence_payload["overview"]["selected_pass_count"] == 4
            and confidence_payload["label_pairwise"]["comparable_pair_count"] == 4
            and math.isclose(confidence_payload["label_pairwise"]["pair_weighted_concordance_pct"], 100.0)
            and confidence_payload["realized_r_pairwise"]["comparable_pair_count"] == 2
            and math.isclose(confidence_payload["realized_r_pairwise"]["pair_weighted_concordance_pct"], 100.0)
            and confidence_payload["semantic_contract"]["extended_candidate_is_not_rescored_as_breakout"] is True
            and (confidence_latest / "audit.md").is_file()
            and (confidence_latest / "competition_pass_candidates.csv").is_file(),
        )

    if strategy_attribution_definition is None:
        raise AssertionError("strategy attribution audit definition missing")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "outputs" / "strategy_compare" / "runs" / "c15_synthetic"
        latest_dir = root / "outputs" / "strategy_compare" / "latest"
        c12_pair = run_dir / "pairs" / "min_roos__all_off__A9__resource_aware_binary_basket"
        c15_pair = run_dir / "pairs" / "min_roos__all_off__CONT12A__resource_aware_continuous"
        c12_pair.mkdir(parents=True, exist_ok=True)
        c15_pair.mkdir(parents=True, exist_ok=True)
        latest_dir.mkdir(parents=True, exist_ok=True)
        synthetic_result = {
            "status": "COMPLETED",
            "config_fingerprint": "synthetic-c15",
            "comparison_period": {"start": "2024-01-01", "end": "2024-01-05"},
            "settings": {
                "dataset": "full",
                "param_policy": "base-finalist-best",
                "max_positions": 10,
                "rotation": "off",
                "arms": {
                    "C3": {"arm_id": "C3", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None},
                    "C12": {"arm_id": "C12", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "A9", "dl_runtime_mode": "resource-aware-binary-basket"},
                    "C15": {"arm_id": "C15", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "CONT12A", "dl_runtime_mode": "resource-aware-continuous"},
                }
            },
            "artifact_identities": {"param:min_roos": {"sha256": "synthetic-param"}},
            "scenarios": {
                "C3": {"total_return_pct": 10.0, "max_drawdown_pct": 8.0, "return_over_max_drawdown": 1.25, "expected_value_r": 1.0, "avg_exposure_pct": 90.0},
                "C12": {"total_return_pct": 12.0, "max_drawdown_pct": 9.0, "return_over_max_drawdown": 1.33, "expected_value_r": 1.2, "avg_exposure_pct": 91.0},
                "C15": {"total_return_pct": 15.0, "max_drawdown_pct": 7.0, "return_over_max_drawdown": 2.14, "expected_value_r": 0.8, "avg_exposure_pct": 90.5},
            },
        }
        (run_dir / "strategy_comparison.json").parent.mkdir(parents=True, exist_ok=True)
        (run_dir / "strategy_comparison.json").write_text(json.dumps(synthetic_result), encoding="utf-8")
        (latest_dir / "manifest.json").write_text(
            json.dumps({"run_dir": "outputs/strategy_compare/runs/c15_synthetic"}), encoding="utf-8"
        )
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]

        def _audit_trade_rows(ticker, realized_r, pnl, invested, reserved, stop, *, signal_date="2024-01-01"):
            return pd.DataFrame([
                {"Date": "2024-01-02", "Ticker": ticker, "Type": "買進 (新訊號)", "買訊日": signal_date, "候選類型": "新訊號", "進場類型": "normal", "成交價": 100.0, "停損價": stop, "股數": 1000, "預留總金額": reserved, "投入總金額": invested},
                {"Date": "2024-01-04", "Ticker": ticker, "Type": "全倉結算(指標)", "成交價": 110.0, "該筆總損益": pnl, "R_Multiple": realized_r},
            ])

        def _audit_capacity(gaps, positions, *, resource=False):
            data = {
                "Date": dates,
                "Post_Execution_Positions": positions,
                "End_Position_Gap": gaps,
                "Filled_Buys_Today": [1, 0, 0],
                "Missed_Buys_Today": [0, 0, 0],
            }
            if resource:
                data.update({
                    "Resource_Aware_Mode": ["dl-selection", "inactive", "inactive"],
                    "Resource_Aware_Changed": [True, False, False],
                    "Resource_Aware_Baseline_Selected": [1, 0, 0],
                    "Resource_Aware_Selected": [2, 0, 0],
                    "Resource_Aware_Baseline_Reserved_Milli": [100000, 0, 0],
                    "Resource_Aware_Reserved_Milli": [110000, 0, 0],
                    "Resource_Aware_Baseline_Score_Sum": [0.5, 0.0, 0.0],
                    "Resource_Aware_Score_Sum": [0.8, 0.0, 0.0],
                    "Resource_Aware_Promoted_Score_Orders": [1, 0, 0],
                    "Resource_Aware_Direct_Score_Order_Feasible": [True, False, False],
                })
            return pd.DataFrame(data)

        def _audit_selected(ticker, *, signal_date="2024-01-01"):
            return pd.DataFrame([{"ticker": ticker, "trade_date": "2024-01-02", "signal_date": signal_date, "type": "買進"}])

        def _audit_equity(values):
            return pd.DataFrame({"Date": dates, "Equity": values})

        _audit_trade_rows("AAA", 1.0, 10000.0, 100000.0, 110000.0, 90.0).to_csv(c15_pair / "no_filter_trades.csv", index=False, encoding="utf-8-sig")
        pd.concat([
            _audit_trade_rows("BBB", 0.8, 12000.0, 80000.0, 100000.0, 88.0),
            _audit_trade_rows("CCC", 1.0, 8000.0, 90000.0, 100000.0, 89.0, signal_date="2023-12-28"),
            _audit_trade_rows("DDD", 0.0, 0.0, 90000.0, 100000.0, 89.0, signal_date="2023-12-27"),
        ], ignore_index=True).to_csv(c15_pair / "score_ranking_trades.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 105.0, 110.0]).to_csv(c15_pair / "no_filter_equity.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 108.0, 115.0]).to_csv(c15_pair / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([1, 1, 0], [9, 9, 10]).to_csv(c15_pair / "no_filter_daily_capacity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([0, 0, 0], [10, 10, 10], resource=True).to_csv(c15_pair / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig")
        _audit_selected("AAA").to_csv(c15_pair / "no_filter_selected_buys.csv", index=False, encoding="utf-8-sig")
        pd.concat([
            _audit_selected("BBB"),
            _audit_selected("CCC", signal_date="2023-12-28"),
            _audit_selected("DDD", signal_date="2023-12-27"),
        ], ignore_index=True).to_csv(c15_pair / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig")
        pd.concat([
            _audit_trade_rows("BBB", 1.2, 11000.0, 95000.0, 105000.0, 89.0, signal_date="2023-12-29"),
            _audit_trade_rows("CCC", 1.0, 10000.0, 95000.0, 105000.0, 89.0, signal_date="2023-12-28"),
            _audit_trade_rows("DDD", 0.0, 0.0, 95000.0, 105000.0, 89.0, signal_date="2023-12-27"),
        ], ignore_index=True).to_csv(c12_pair / "score_ranking_trades.csv", index=False, encoding="utf-8-sig")
        _audit_equity([100.0, 106.0, 112.0]).to_csv(c12_pair / "score_ranking_equity.csv", index=False, encoding="utf-8-sig")
        _audit_capacity([1, 0, 0], [9, 10, 10], resource=True).to_csv(c12_pair / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig")
        pd.concat([
            _audit_selected("BBB", signal_date="2023-12-29"),
            _audit_selected("CCC", signal_date="2023-12-28"),
            _audit_selected("DDD", signal_date="2023-12-27"),
        ], ignore_index=True).to_csv(c12_pair / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig")

        c15_definition = type(strategy_attribution_definition)(
            module_id=strategy_attribution_definition.module_id,
            audit_id=strategy_attribution_definition.audit_id,
            enabled=True,
            audit_type=strategy_attribution_definition.audit_type,
            description=strategy_attribution_definition.description,
            source={"kind": "strategy_compare", "run": "latest", "candidate_arm_id": "C15", "comparator_arm_ids": ["C3", "C12"]},
            dimensions={"focus_year": 2024, "top_month_count": 2, "top_trade_count": 5},
            outcomes={**dict(strategy_attribution_definition.outcomes), "risk_dollar_translation": True},
            output_subdir="breakout_quality/c15_strategy_attribution_synthetic",
        )
        c15_status = collect_strategy_attribution_status(c15_definition, project_root=root)
        c15_payload = run_strategy_attribution_audit(c15_definition, project_root=root, quiet=True)
        c15_latest = root / Path(AUDIT_OUTPUT_ROOT) / c15_definition.output_subdir / "latest"
        exact_paths = []
        for comparison in c15_payload["comparisons"]:
            daily = pd.read_csv(
                c15_latest / f"C15_vs_{comparison['comparator_arm_id']}_daily_log_wealth.csv",
                encoding="utf-8-sig",
            )
            exact_paths.append(
                math.isclose(
                    float(daily["delta_log_wealth"].sum()),
                    float(comparison["wealth_path"]["target_delta_log_wealth"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        c15_vs_c3 = next(
            item for item in c15_payload["comparisons"]
            if item["comparator_arm_id"] == "C3"
        )
        c15_vs_c12 = next(
            item for item in c15_payload["comparisons"]
            if item["comparator_arm_id"] == "C12"
        )
        c15_report_text = (c15_latest / "audit.md").read_text(encoding="utf-8")
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "c15_attribution_is_read_only_exact_wealth_path_and_cross_arm_capital_geometry_audit",
            True,
            c15_status["status"] == "READY"
            and c15_payload["schema_version"] == 3
            and [item["comparator_arm_id"] for item in c15_payload["comparisons"]] == ["C3", "C12"]
            and all(exact_paths)
            and all(item["selection"]["changed_days"] == 1 for item in c15_payload["comparisons"])
            and c15_vs_c12["trade_contribution"]["common_trade_count"] == 3
            and c15_vs_c12["trade_contribution"]["candidate_only_trade_count"] == 0
            and c15_vs_c12["trade_contribution"]["comparator_only_trade_count"] == 0
            and c15_vs_c12["risk_dollar_translation"]["common"]["risk_covered_trade_count"] == 2
            and math.isclose(c15_vs_c12["risk_dollar_translation"]["common"]["risk_coverage_pct"], 200.0 / 3.0, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(c15_vs_c12["risk_dollar_translation"]["common"]["risk_size_effect_pnl"], 3833.333333333333, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(c15_vs_c12["risk_dollar_translation"]["common"]["r_difference_effect_pnl"], -4833.333333333333, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(c15_vs_c12["risk_dollar_translation"]["common"]["decomposition_residual_pnl"], 0.0, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(
                c15_vs_c3["concentration"]["non_focus_delta_log_wealth"],
                c15_vs_c3["concentration"]["net_delta_log_wealth"]
                - c15_vs_c3["concentration"]["focus_year_delta_log_wealth"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and c15_vs_c3["slot_occupancy"]["candidate_resource_aware_selected_order_delta"] == 1
            and c15_vs_c3["slot_occupancy"]["candidate_resource_aware_reserved_delta_milli"] == 10000
            and c15_vs_c3["slot_occupancy"]["candidate_resource_aware_promoted_score_orders"] == 1
            and c15_vs_c3["slot_occupancy"]["candidate_resource_aware_direct_score_order_days"] == 1
            and "Exclusive selection ΔPnL" in c15_report_text
            and "All trade ΔPnL" in c15_report_text
            and "Exclusive risk-dollar translation" in c15_report_text
            and "Risk-size/path effect" in c15_report_text
            and "此表只比較C15 selector" in c15_report_text
            and "非2024期間相對wealth effect" in c15_report_text
            and "C15 selector自身盤前診斷" in c15_report_text
            and c15_payload["metadata"]["read_only"] is True
            and c15_payload["metadata"]["portfolio_replay_executed"] is False
            and (c15_latest / "audit.md").is_file()
            and (c15_latest / "audit.json").is_file(),
        )

        if source_attribution_definition is None:
            raise AssertionError("缺少c15-source-attribution正式設定")
        c14_run_dir = root / "outputs" / "strategy_compare" / "runs" / "c14_synthetic"
        c14_pair = c14_run_dir / "pairs" / "min_roos__all_off__CONT11G__resource_aware_continuous"
        c14_pair.mkdir(parents=True, exist_ok=True)
        c14_result = {
            "status": "COMPLETED",
            "config_fingerprint": "synthetic-c14",
            "comparison_period": {"start": "2024-01-01", "end": "2024-01-05"},
            "settings": {
                "dataset": "full",
                "param_policy": "base-finalist-best",
                "max_positions": 10,
                "rotation": "off",
                "arms": {
                    "C14": {"arm_id": "C14", "enabled": True, "param_source": "min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "CONT11G", "dl_runtime_mode": "resource-aware-continuous"},
                },
            },
            "artifact_identities": {"param:min_roos": {"sha256": "synthetic-param"}},
            "scenarios": {
                "C14": {"total_return_pct": 13.0, "max_drawdown_pct": 8.0, "return_over_max_drawdown": 1.625, "expected_value_r": 0.7, "avg_exposure_pct": 90.4},
            },
        }
        (c14_run_dir / "strategy_comparison.json").write_text(
            json.dumps(c14_result), encoding="utf-8"
        )
        _audit_trade_rows("CCC", 0.7, 9000.0, 85000.0, 98000.0, 87.0).to_csv(
            c14_pair / "score_ranking_trades.csv", index=False, encoding="utf-8-sig"
        )
        _audit_equity([100.0, 107.0, 113.0]).to_csv(
            c14_pair / "score_ranking_equity.csv", index=False, encoding="utf-8-sig"
        )
        _audit_capacity([1, 0, 0], [9, 10, 10], resource=True).to_csv(
            c14_pair / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig"
        )
        _audit_selected("CCC").to_csv(
            c14_pair / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig"
        )
        source_definition = type(source_attribution_definition)(
            module_id=source_attribution_definition.module_id,
            audit_id=source_attribution_definition.audit_id,
            enabled=True,
            audit_type=source_attribution_definition.audit_type,
            description=source_attribution_definition.description,
            source={
                "kind": "strategy_compare",
                "candidate_arm_id": "C15",
                "comparator_arm_ids": ["C14"],
                "arm_runs": {
                    "C15": {"config_fingerprint": "synthetic-c15"},
                    "C14": {"config_fingerprint": "synthetic-c14"},
                },
            },
            dimensions={"focus_year": 2024, "top_month_count": 2, "top_trade_count": 5},
            outcomes=dict(source_attribution_definition.outcomes),
            output_subdir="breakout_quality/c15_source_attribution_synthetic",
        )
        source_status = collect_strategy_attribution_status(source_definition, project_root=root)
        source_payload = run_strategy_attribution_audit(source_definition, project_root=root, quiet=True)
        bad_c14_result = json.loads(json.dumps(c14_result))
        bad_c14_result["artifact_identities"]["param:min_roos"]["sha256"] = "different-param"
        (c14_run_dir / "strategy_comparison.json").write_text(
            json.dumps(bad_c14_result), encoding="utf-8"
        )
        bad_source_status = collect_strategy_attribution_status(source_definition, project_root=root)
        (c14_run_dir / "strategy_comparison.json").write_text(
            json.dumps(c14_result), encoding="utf-8"
        )
        source_pair = source_payload["comparisons"][0]
        source_latest = root / Path(AUDIT_OUTPUT_ROOT) / source_definition.output_subdir / "latest"
        source_daily = pd.read_csv(
            source_latest / "C15_vs_C14_daily_log_wealth.csv", encoding="utf-8-sig"
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "c15_source_attribution_reuses_compatible_completed_runs_by_fingerprint_without_replay",
            True,
            source_status["status"] == "READY"
            and source_status["source"]["cross_run"] is True
            and bad_source_status["status"] == "BLOCKED"
            and "策略參數工件不一致" in bad_source_status["reason"]
            and source_payload["schema_version"] == 3
            and source_payload["metadata"]["cross_run"] is True
            and source_payload["metadata"]["strategy_compare_config_fingerprints"] == {"C15": "synthetic-c15", "C14": "synthetic-c14"}
            and source_pair["comparator_arm_id"] == "C14"
            and "comparator_resource_aware_changed_days" in source_pair["slot_occupancy"]
            and math.isclose(
                float(source_daily["delta_log_wealth"].sum()),
                float(source_pair["wealth_path"]["target_delta_log_wealth"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and source_payload["metadata"]["portfolio_replay_executed"] is False,
        )

        if pit_realization_definition is None:
            raise AssertionError("缺少c23-c25-pit-realization正式設定")
        pit_run = root / "outputs" / "strategy_compare" / "runs" / "pit_realization_synthetic"
        pit_pair_24 = pit_run / "pairs" / "selection_min_roos__all_off__CONT12B_PIT__resource_aware_continuous_max_dl"
        pit_pair_25 = pit_run / "pairs" / "selection_min_roos__all_off__CONT12B_PIT__resource_aware_continuous_max_dl_feasible_ascent"
        pit_pair_24.mkdir(parents=True, exist_ok=True)
        pit_pair_25.mkdir(parents=True, exist_ok=True)
        pit_result = {
            "status": "COMPLETED",
            "config_fingerprint": "synthetic-pit-realization",
            "comparison_period": {"start": "2024-01-01", "end": "2024-01-05"},
            "settings": {
                "dataset": "full",
                "param_policy": "base-finalist-best",
                "max_positions": 10,
                "rotation": "off",
                "arms": {
                    "C23": {"arm_id": "C23", "enabled": True, "param_source": "selection_min_roos", "rule_policy": "all_off", "dl_enabled": False, "dl_id": None, "dl_runtime_mode": None},
                    "C24": {"arm_id": "C24", "enabled": True, "param_source": "selection_min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "CONT12B_PIT", "dl_runtime_mode": "resource-aware-continuous-max-dl"},
                    "C25": {"arm_id": "C25", "enabled": True, "param_source": "selection_min_roos", "rule_policy": "all_off", "dl_enabled": True, "dl_id": "CONT12B_PIT", "dl_runtime_mode": "resource-aware-continuous-max-dl-feasible-ascent"},
                },
            },
            "scenarios": {
                "C23": {"total_return_pct": 12.0, "max_drawdown_pct": 8.0, "return_over_max_drawdown": 1.5, "expected_value_r": 1.0, "avg_exposure_pct": 90.0},
                "C24": {"total_return_pct": 8.0, "max_drawdown_pct": 8.0, "return_over_max_drawdown": 1.0, "expected_value_r": 0.4, "avg_exposure_pct": 82.0},
                "C25": {"total_return_pct": 10.0, "max_drawdown_pct": 8.5, "return_over_max_drawdown": 1.1765, "expected_value_r": 0.7, "avg_exposure_pct": 86.0},
            },
        }
        (pit_run / "strategy_comparison.json").write_text(json.dumps(pit_result), encoding="utf-8")

        def _write_pit_pair(pair_dir, *, ticker, realized_r, pnl, invested, reserved, stop, target_r, return_pct, romd, ev, gap):
            _audit_trade_rows("AAA", 1.0, 10000.0, 100000.0, 110000.0, 90.0).to_csv(
                pair_dir / "no_filter_trades.csv", index=False, encoding="utf-8-sig"
            )
            _audit_trade_rows(ticker, realized_r, pnl, invested, reserved, stop).to_csv(
                pair_dir / "score_ranking_trades.csv", index=False, encoding="utf-8-sig"
            )
            _audit_equity([100.0, 105.0, 112.0]).to_csv(
                pair_dir / "no_filter_equity.csv", index=False, encoding="utf-8-sig"
            )
            _audit_equity([100.0, 103.0, 108.0 if ticker == "BBB" else 110.0]).to_csv(
                pair_dir / "score_ranking_equity.csv", index=False, encoding="utf-8-sig"
            )
            _audit_capacity([1, 1, 0], [9, 9, 10]).to_csv(
                pair_dir / "no_filter_daily_capacity.csv", index=False, encoding="utf-8-sig"
            )
            _audit_capacity([gap, gap, 0], [10-gap, 10-gap, 10], resource=True).to_csv(
                pair_dir / "score_ranking_daily_capacity.csv", index=False, encoding="utf-8-sig"
            )
            _audit_selected("AAA").to_csv(
                pair_dir / "no_filter_selected_buys.csv", index=False, encoding="utf-8-sig"
            )
            _audit_selected(ticker).to_csv(
                pair_dir / "score_ranking_selected_buys.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{
                "ticker": "AAA", "trade_date": "2024-01-02", "signal_date": "2024-01-01",
                "score_event_date": "2024-01-01", "target_raw_r": 0.8,
            }]).to_csv(pair_dir / "no_filter_selected_target_diagnostics.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame([{
                "ticker": ticker, "trade_date": "2024-01-02", "signal_date": "2024-01-01",
                "score_event_date": "2024-01-01", "target_raw_r": target_r,
            }]).to_csv(pair_dir / "score_ranking_selected_target_diagnostics.csv", index=False, encoding="utf-8-sig")
            pair_payload = {
                "metadata": {
                    "comparison_mode": "score-ranking",
                    "comparison_period": {"start": "2024-01-01", "end": "2024-01-05"},
                    "score_source": "selection_point_in_time",
                },
                "no_filter": pit_result["scenarios"]["C23"],
                "score_ranking": {
                    "total_return_pct": return_pct, "max_drawdown_pct": 8.0,
                    "return_over_max_drawdown": romd, "expected_value_r": ev,
                    "avg_exposure_pct": 82.0 if ticker == "BBB" else 86.0,
                },
                "selection_diagnostics": {
                    "score_ranking_minus_no_filter": {
                        "selected_target_mean_r": target_r - 0.8,
                        "target_top_k_retention_mean": 0.01,
                    }
                },
            }
            (pair_dir / "strategy_comparison.json").write_text(json.dumps(pair_payload), encoding="utf-8")

        _write_pit_pair(
            pit_pair_24, ticker="BBB", realized_r=0.4, pnl=4000.0, invested=70000.0,
            reserved=100000.0, stop=85.0, target_r=1.4, return_pct=8.0, romd=1.0, ev=0.4, gap=2,
        )
        _write_pit_pair(
            pit_pair_25, ticker="CCC", realized_r=0.7, pnl=7000.0, invested=85000.0,
            reserved=105000.0, stop=88.0, target_r=1.2, return_pct=10.0, romd=1.1765, ev=0.7, gap=1,
        )
        pit_definition = type(pit_realization_definition)(
            module_id=pit_realization_definition.module_id,
            audit_id=pit_realization_definition.audit_id,
            enabled=True,
            audit_type=pit_realization_definition.audit_type,
            description=pit_realization_definition.description,
            source={
                "kind": "strategy_compare",
                "run": "outputs/strategy_compare/runs/pit_realization_synthetic",
                "baseline_arm_id": "C23",
                "candidate_arm_ids": ["C24", "C25"],
            },
            dimensions={"focus_year": 2024, "top_month_count": 2, "top_trade_count": 5},
            outcomes=dict(pit_realization_definition.outcomes),
            output_subdir="breakout_quality/c23_c25_pit_realization_synthetic",
        )
        pit_status = collect_strategy_realization_capture_status(pit_definition, project_root=root)
        pit_payload = run_strategy_realization_capture_audit(pit_definition, project_root=root, quiet=True)
        pit_latest = root / Path(AUDIT_OUTPUT_ROOT) / pit_definition.output_subdir / "latest"
        pit_report = (pit_latest / "audit.md").read_text(encoding="utf-8")
        c24_pair = next(item for item in pit_payload["comparisons"] if item["candidate_arm_id"] == "C24")
        c25_pair = next(item for item in pit_payload["comparisons"] if item["candidate_arm_id"] == "C25")
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "selection_pit_realization_capture_audit_is_read_only_config_driven_and_exposes_target_to_realized_gap",
            True,
            pit_status["status"] == "READY"
            and [item["candidate_arm_id"] for item in pit_payload["comparisons"]] == ["C24", "C25"]
            and c24_pair["interpretation"]["target_to_realized_divergence"] is True
            and c25_pair["interpretation"]["target_to_realized_divergence"] is True
            and c24_pair["structural"]["trade_contribution"]["exclusive_selection_delta_r"] < 0
            and c25_pair["structural"]["trade_contribution"]["exclusive_selection_delta_r"] < 0
            and math.isclose(
                float(c24_pair["exclusive_trade_breakdown"]["implied_selection_delta_r"]),
                float(c24_pair["structural"]["trade_contribution"]["exclusive_selection_delta_r"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(c25_pair["exclusive_trade_breakdown"]["implied_selection_delta_r"]),
                float(c25_pair["structural"]["trade_contribution"]["exclusive_selection_delta_r"]),
                rel_tol=0.0,
                abs_tol=1e-8,
            )
            and c24_pair["exclusive_trade_breakdown"]["primary_realized_driver"] in {"winner_capture", "loser_avoidance", "balanced", "mixed"}
            and c25_pair["exclusive_trade_breakdown"]["primary_realized_driver"] in {"winner_capture", "loser_avoidance", "balanced", "mixed"}
            and c24_pair["capture"]["score_sort"]["avg_target_r"] > c24_pair["capture"]["baseline"]["avg_target_r"]
            and c24_pair["capture"]["score_sort"]["avg_realized_r"] < c24_pair["capture"]["baseline"]["avg_realized_r"]
            and c24_pair["structural"]["slot_occupancy"]["candidate_end_position_gap_slot_days"] > c24_pair["structural"]["slot_occupancy"]["comparator_end_position_gap_slot_days"]
            and pit_payload["metadata"]["portfolio_replay_executed"] is False
            and pit_payload["metadata"]["training_performed"] is False
            and "Target → Realized R / 資金捕捉" in pit_report
            and "Exclusive trades" in pit_report
            and "Exclusive R分解" in pit_report
            and (pit_latest / "audit.json").is_file()
            and (pit_latest / "C24_vs_C23_capture_candidate_lifecycle.csv").is_file()
            and (pit_latest / "C25_vs_C23_trade_contributions.csv").is_file(),
        )

    app_source = (Path(__file__).resolve().parents[2] / "tools" / "filters" / "breakout_quality" / "application.py").read_text(encoding="utf-8")
    project_audit_source = (Path(__file__).resolve().parents[2] / "apps" / "research.py").read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "project_audit_entry_and_breakout_quality_facade_share_one_config_driven_backend",
        True,
        "get_domain_cli_commands" in app_source
        and "Audit／診斷" not in app_source[app_source.find("def _interactive_model_research"):app_source.find("def run_model_training_menu")]
        and _source_has_render_menu_item_call(
            project_audit_source,
            index=4,
            label="Audit／診斷",
        )
        and "get_active_audit_module_id" in project_audit_source
        and "get_audit_module_ids" not in project_audit_source
        and "run_enabled_audits" in project_audit_source,
    )

    summary["enabled_audit_ids"] = [item.audit_id for item in enabled_definitions]
    return results, summary
