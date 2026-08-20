from __future__ import annotations

from .synthetic_breakout_quality_support import (
    BREAKOUT_OPTIMIZER_SEARCH_SPACE,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
    BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
    BUY_LIMIT_OVERAGE_SORT_METHOD,
    COMPARISON_MODE_HARD_FILTER,
    COMPARISON_MODE_SCORE_RANKING,
    LABEL_INVALID,
    LABEL_PASS,
    LABEL_REJECT,
    OPTIONAL_ENTRY_FILTER_FIELDS,
    OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    PARAM_POLICY_BASE_FINALISTS_AGREE,
    PARAM_POLICY_BASE_FINALIST_BEST,
    Path,
    SCORE_COLUMN,
    SimpleNamespace,
    UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    V16StrategyParams,
    _aggregate_ensemble_candidate_rows,
    _assert_controlled_ensemble_pair,
    _assert_controlled_param_pair,
    _build_controlled_param_source_pair,
    _capacity_summary,
    _comparison_labels,
    _comparison_output_dir_name,
    _first_existing_comparison_dir,
    _iter_reentry_watch_targets,
    _load_param_source,
    _normalize_yearly_completeness,
    _resolve_candidate_quality_ranking,
    _resolve_comparison_period,
    _resolve_params_path,
    _source_has_render_menu_item_call,
    _to_json_native,
    _validate_requested_param_policy,
    add_check,
    build_static_active_param_ensemble_payload,
    build_trade_attribution,
    calc_projected_capital_metrics,
    create_breakout_reentry_signal_state,
    create_breakout_reentry_watch_state,
    io,
    json,
    np,
    os,
    params_to_json_dict,
    patch,
    pd,
    redirect_stdout,
    replace,
    resolve_breakout_quality_rank,
    resolve_filter_artifact_paths,
    sort_candidate_rows,
    subprocess,
    sys,
    tempfile,
)

def validate_breakout_quality_strategy_comparison_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STRATEGY_COMPARISON"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    base = V16StrategyParams()
    common = {
        "breakout_quality_filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "breakout_quality_score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
    }
    no_filter = replace(base, use_breakout_quality_filter=False, **common)
    quality_filter = replace(base, use_breakout_quality_filter=True, **common)
    try:
        _assert_controlled_param_pair(no_filter, quality_filter)
        single_switch_only = True
    except ValueError:
        single_switch_only = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "controlled_pair_only_toggles_quality_filter",
        True,
        single_switch_only,
    )

    try:
        _assert_controlled_param_pair(
            no_filter,
            replace(quality_filter, high_len=int(quality_filter.high_len) + 1),
        )
        extra_difference_rejected = False
    except ValueError as exc:
        extra_difference_rejected = "high_len" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "additional_param_difference_is_rejected",
        True,
        extra_difference_rejected,
    )

    synthetic_period_contract = SimpleNamespace(
        execution_start=pd.Timestamp("2025-01-03").date(),
        required_signal_start=pd.Timestamp("2025-01-02").date(),
        available_from=pd.Timestamp("2025-01-02").date(),
        available_through=pd.Timestamp("2025-12-31").date(),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_comparison_starts_at_execution_window_not_signal_score_anchor",
        ("2025-01-03", "2025-12-31"),
        _resolve_comparison_period(synthetic_period_contract),
    )

    ranking_pair = _build_controlled_param_source_pair(
        {"kind": "single_param", "params": base},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
    )
    ranking_left = params_to_json_dict(ranking_pair[1])
    ranking_right = params_to_json_dict(ranking_pair[2])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_pair_only_toggles_ranking_and_keeps_hard_filter_off",
        (False, False, False, True),
        (
            ranking_left["use_breakout_quality_filter"],
            ranking_right["use_breakout_quality_filter"],
            ranking_left["use_breakout_quality_ranking"],
            ranking_right["use_breakout_quality_ranking"],
        ),
    )

    all_off_pair = _build_controlled_param_source_pair(
        {"kind": "single_param", "params": base},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    all_off_left = params_to_json_dict(all_off_pair[1])
    all_off_right = params_to_json_dict(all_off_pair[2])
    from contextlib import nullcontext
    breakout_quality_app = __import__(
        "tools.filters.breakout_quality.application", fromlist=["*"]
    )
    from config import breakout_quality as breakout_quality_config

    with patch.object(
        breakout_quality_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ):
        binary_menu_settings = (
            breakout_quality_config.get_breakout_quality_workflow_settings()
        )
    export_request = SimpleNamespace(
        filter_id=binary_menu_settings.filter_id,
        experiment_profile=binary_menu_settings.experiment_profile,
        evaluation_batch_size=4096,
        evaluation_workers=4,
        device="auto",
        mixed_precision_dtype="auto",
        mixed_precision=True,
        deterministic_algorithms=True,
        allow_tf32=False,
        preload_feature_bank=True,
    )
    post_train_calls = []

    def _capture_post_train_command(command, command_args, *, program_name):
        post_train_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_post_train_command,
        ),
        patch.object(
            breakout_quality_app,
            "_compact_console_scope",
            side_effect=lambda: nullcontext(),
        ),
        redirect_stdout(io.StringIO()),
    ):
        post_train_rc = breakout_quality_app._run_binary_post_train_validation(
            export_request,
            workflow_settings=binary_menu_settings,
            program_name="apps/research.py model",
        )
    export_call = post_train_calls[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_model_post_train_exports_forward_scores_without_strategy_replay",
        (0, ("export-scores",), True, True),
        (
            post_train_rc,
            tuple(call[0] for call in post_train_calls),
            "forward_oos" in export_call[1]
            and binary_menu_settings.experiment_profile in export_call[1],
            "strategy-compare" not in tuple(call[0] for call in post_train_calls),
        ),
    )

    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "tools" / "filters" / "breakout_quality" / "application.py").read_text(
        encoding="utf-8"
    )
    trade_path_train_source = app_source.split(
        "def _interactive_trade_path_train_and_report", 1
    )[1].split("def _interactive_trade_path_existing_report", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_menu_training_flow_builds_trade_path_label_and_complete_model_artifacts_only",
        True,
        all(
            token in trade_path_train_source
            for token in (
                "build-trade-path-labels",
                '"train"',
                '"export-scores"',
                '"report"',
                'scope="forward_oos"',
                "本流程不執行策略績效比較",
            )
        )
        and "_run_binary_post_train_validation" not in trade_path_train_source
        and "strategy-compare" not in trade_path_train_source,
    )

    report_calls = []

    def _capture_trade_path_report(command, command_args, *, program_name):
        report_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_trade_path_report,
        ),
        redirect_stdout(io.StringIO()),
    ):
        trade_path_report_rc = breakout_quality_app._run_trade_path_model_report(
            "apps/research.py model",
            request=export_request,
            export_research_scores=True,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "existing_trade_path_model_report_exports_research_and_forward_scores_without_replay",
        (0, ("export-scores", "report", "export-scores"), True, True),
        (
            trade_path_report_rc,
            tuple(call[0] for call in report_calls),
            "research" in report_calls[0][1]
            and "forward_oos" in report_calls[2][1],
            "strategy-compare" not in tuple(call[0] for call in report_calls),
        ),
    )

    existing_binary_source = app_source.split(
        "def _interactive_trade_path_existing_report", 1
    )[1].split("def _interactive_trade_path_label_summary", 1)[0]
    binary_menu_source = app_source.split(
        "def _interactive_binary_model_research", 1
    )[1].split("def _interactive_model_research", 1)[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_model_menu_supports_trade_path_existing_model_report_without_strategy_replay",
        True,
        all(
            token in binary_menu_source
            for token in (
                "建立新Label",
                "使用既有模型",
                "查看Label與事件生命週期摘要",
                "_interactive_trade_path_existing_report",
            )
        )
        and "_run_trade_path_model_report" in existing_binary_source
        and "strategy-compare" not in existing_binary_source
        and "_run_binary_post_train_validation" not in existing_binary_source,
    )

    existing_model_calls = []

    def _capture_existing_model_command(command, command_args, *, program_name):
        existing_model_calls.append((command, tuple(command_args), program_name))
        return 0

    with (
        patch.object(
            breakout_quality_app,
            "_run_command",
            side_effect=_capture_existing_model_command,
        ),
        patch.object(
            breakout_quality_app,
            "_prompt_bool",
            return_value=True,
        ),
        patch.object(
            breakout_quality_app,
            "_print_policy_defaults",
            return_value=None,
        ),
        patch.object(
            breakout_quality_app,
            "_compact_console_scope",
            side_effect=lambda: nullcontext(),
        ),
        redirect_stdout(io.StringIO()),
    ):
        existing_model_rc = (
            breakout_quality_app._interactive_trade_path_existing_report(
                "apps/research.py model",
                workflow_settings=binary_menu_settings,
            )
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_existing_trade_path_model_menu_completes_research_report_and_forward_scores_only",
        (
            0,
            ("export-scores", "report", "export-scores"),
            True,
            True,
        ),
        (
            existing_model_rc,
            tuple(call[0] for call in existing_model_calls),
            "research" in existing_model_calls[0][1]
            and "forward_oos" in existing_model_calls[2][1],
            "--include-oos" in existing_model_calls[1][1]
            and "strategy-compare" not in tuple(
                call[0] for call in existing_model_calls
            ),
        ),
    )

    # Regression: hard-filter comparison has no score-ranking capture audit.
    # The shared console renderer must receive None rather than an unbound local.
    from datetime import date as _date
    from filters.breakout_quality import strategy_compare_engine as strategy_compare_module

    with tempfile.TemporaryDirectory() as hard_filter_tmp_dir:
        hard_filter_root = Path(hard_filter_tmp_dir)
        hard_filter_params = hard_filter_root / "params.json"
        hard_filter_params.write_text("{}\n", encoding="utf-8")
        hard_filter_runtime = SimpleNamespace(
            execution_start=_date(2021, 1, 1),
            required_signal_start=_date(2020, 12, 31),
            available_from=_date(2020, 12, 31),
            available_through=_date(2021, 12, 31),
            manifest={
                "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                "experiment_profile": BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                "fixed_evaluation_threshold": float(
                    BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD
                ),
                "runtime_eligibility": {},
                "score_table": {},
            },
            paths=SimpleNamespace(
                manifest_path=hard_filter_root / "manifest.json",
                score_path=hard_filter_root / "scores.csv",
            ),
        )
        hard_filter_profile = {
            "portfolio_capacity_rows": [],
            "closed_trade_rows": [],
        }
        hard_filter_payload = {
            "profile": hard_filter_profile,
            "trade_history": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
        }
        hard_filter_summary = {
            "total_return_pct": 1.0,
            "max_drawdown_pct": 1.0,
            "return_over_max_drawdown": 1.0,
            "trade_count": 0,
        }
        hard_filter_console_capture = []

        def _capture_hard_filter_console(*args, **kwargs):
            hard_filter_console_capture.append(kwargs.get("capture_result"))
            return "hard-filter-report"

        with (
            patch.object(
                strategy_compare_module,
                "load_runtime_artifact_contract",
                return_value=hard_filter_runtime,
            ),
            patch.object(
                strategy_compare_module,
                "_resolve_params_path",
                return_value=hard_filter_params,
            ),
            patch.object(
                strategy_compare_module,
                "_load_param_source",
                return_value={"kind": "single_param"},
            ),
            patch.object(
                strategy_compare_module,
                "_validate_requested_param_policy",
                return_value={
                    "selector": "single_param",
                    "member_count_min": 1,
                    "member_count_max": 1,
                    "min_agree": 1,
                },
            ),
            patch.object(
                strategy_compare_module,
                "_build_controlled_param_source_pair",
                return_value=(
                    "single_param",
                    base,
                    quality_filter,
                    {},
                    {},
                    None,
                ),
            ),
            patch.object(
                strategy_compare_module,
                "get_dataset_dir",
                return_value=str(hard_filter_root),
            ),
            patch.object(
                strategy_compare_module,
                "_run_scenario",
                side_effect=[hard_filter_payload, hard_filter_payload],
            ),
            patch.object(strategy_compare_module, "_assert_shared_benchmark"),
            patch.object(
                strategy_compare_module,
                "_scenario_summary",
                side_effect=[hard_filter_summary, hard_filter_summary],
            ),
            patch.object(
                strategy_compare_module,
                "_build_yearly_comparison",
                return_value=pd.DataFrame(
                    columns=[
                        "year",
                        "no_filter_return_pct",
                        "quality_filter_return_pct",
                        "is_full_year",
                    ]
                ),
            ),
            patch.object(
                strategy_compare_module,
                "_markdown_report",
                return_value="# hard-filter\n",
            ),
            patch.object(strategy_compare_module, "_remove_legacy_html_outputs"),
            patch.object(strategy_compare_module, "write_trade_attribution_outputs"),
            patch.object(
                strategy_compare_module,
                "_render_strategy_console_report",
                side_effect=_capture_hard_filter_console,
            ),
            patch.object(strategy_compare_module, "print_artifact_paths"),
            redirect_stdout(io.StringIO()),
        ):
            hard_filter_result = strategy_compare_module.run_comparison(
                project_root=hard_filter_root,
                dataset="full",
                params_path=hard_filter_params,
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                max_positions=10,
                enable_rotation=False,
                allow_static_diagnostic=True,
                comparison_mode=COMPARISON_MODE_HARD_FILTER,
                filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                score_source=strategy_compare_module.SCORE_SOURCE_CANONICAL_RUNTIME,
                model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
                experiment_profile=BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
                output_dir_override=hard_filter_root / "comparison",
                quiet=False,
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_strategy_compare_passes_none_capture_audit_to_shared_console",
        (None, True),
        (
            hard_filter_console_capture[0]
            if hard_filter_console_capture
            else "console-not-called",
            "quality_filter" in hard_filter_result,
        ),
    )

    all_off_output_name = _comparison_output_dir_name(
        COMPARISON_MODE_SCORE_RANKING,
        _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
        param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
        ranking_policy="score",
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "canonical_all_off_score_ranking_keeps_optional_filters_off_and_isolates_output",
        (
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            True,
        ),
        (
            tuple(all_off_left[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            tuple(all_off_right[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            (
                all_off_left["use_breakout_quality_ranking"],
                all_off_right["use_breakout_quality_ranking"],
            ),
            all_off_output_name.endswith("_optional_entry_filters_all_off"),
        ),
    )





    baseline_sort_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": False},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": False},
    ]
    ranking_sort_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.90},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.40},
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "single_param_score_ranking_precedes_existing_overage_sort",
        (["B", "A"], ["A", "B"]),
        (
            [row["ticker"] for row in sort_candidate_rows(baseline_sort_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
            [row["ticker"] for row in sort_candidate_rows(ranking_sort_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
        ),
    )

    projected_fraction, deployment_rate = calc_projected_capital_metrics(
        proj_cost=150.0,
        sizing_capital=1000.0,
        max_position_cap_pct=0.30,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_deployment_uses_canonical_projected_cost_and_position_cap",
        (0.15, 0.50),
        (round(projected_fraction, 6), round(deployment_rate, 6)),
    )

    capital_rows = [
        {
            "ticker": "A", "sort_value": 0.05, "proj_cost": 100.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 1.0 / 3.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.90,
            "breakout_quality_rank": {"available": True},
        },
        {
            "ticker": "B", "sort_value": 0.10, "proj_cost": 200.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 2.0 / 3.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.78,
            "breakout_quality_rank": {"available": True},
        },
        {
            "ticker": "C", "sort_value": 0.20, "proj_cost": 300.0,
            "sizing_capital": 1000.0, "max_position_cap_pct": 0.30,
            "projected_capital_deployment_rate": 1.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_score": 0.68,
            "breakout_quality_rank": {"available": True},
        },
    ]
    r2_rows = [
        dict(row, breakout_quality_ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED)
        for row in capital_rows
    ]
    r3_rows = [
        dict(row, breakout_quality_ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET)
        for row in capital_rows
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_adjusted_score_multiplies_score_by_formal_deployment_rate",
        ["C", "B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(r2_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_bucket_policy_uses_daily_tercile_then_score",
        ["C", "B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(r3_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)
        ],
    )

    missing_capital_rows = [
        {
            "ticker": "A", "sort_value": 0.20, "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            "breakout_quality_score": None,
            "breakout_quality_rank": {"available": False},
        },
        {
            "ticker": "B", "sort_value": 0.10, "proj_cost": 80.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            "breakout_quality_score": None,
            "breakout_quality_rank": {"available": False},
        },
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_aware_missing_scores_preserve_original_buy_sort_fallback",
        ["B", "A"],
        [
            row["ticker"]
            for row in sort_candidate_rows(
                missing_capital_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD
            )
        ],
    )

    ensemble_rank_rows = []
    for ticker, votes, score, overage in (("A", 6, 0.20, 0.0), ("B", 5, 0.99, 0.0), ("C", 6, 0.80, 1.0)):
        for member_idx in range(votes):
            ensemble_rank_rows.append({
                "ticker": ticker,
                "ensemble_member_key": f"m{member_idx}",
                "params_obj": base,
                "sort_value": overage,
                "proj_cost": 100.0,
                "use_breakout_quality_ranking": True,
                "breakout_quality_score": score,
                "breakout_quality_score_date": f"2025-01-{member_idx + 2:02d}",
                "breakout_quality_rank": {
                    "score": score,
                    "available": True,
                    "unavailable_reason": "",
                    "score_date": f"2025-01-{member_idx + 2:02d}",
                    "shared_group_score": True,
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                },
            })
    ensemble_ranked = _aggregate_ensemble_candidate_rows(ensemble_rank_rows, min_agree=3)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_votes_remain_first_and_score_only_reorders_equal_vote_candidates",
        [("C", 6), ("A", 6), ("B", 5)],
        [(row["ticker"], row["ensemble_vote_count"]) for row in ensemble_ranked],
    )


    ensemble_capital_rows = []
    for ticker, votes, score, deployment_rate in (
        ("A", 2, 0.99, 0.10),
        ("B", 2, 0.80, 0.80),
        ("C", 3, 0.10, 1.00),
    ):
        for member_idx in range(votes):
            ensemble_capital_rows.append({
                "ticker": ticker,
                "ensemble_member_key": f"m{member_idx}",
                "params_obj": base,
                "sort_value": 0.0,
                "proj_cost": 100.0,
                "sizing_capital": 1000.0,
                "max_position_cap_pct": 0.30,
                "projected_capital_fraction": deployment_rate * 0.30,
                "projected_capital_deployment_rate": deployment_rate,
                "use_breakout_quality_ranking": True,
                "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
                "breakout_quality_score": score,
                "breakout_quality_score_date": "2025-01-02",
                "breakout_quality_rank": {
                    "score": score,
                    "available": True,
                    "unavailable_reason": "",
                    "score_date": "2025-01-02",
                    "score_source": "selection_point_in_time",
                    "shared_group_score": True,
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                },
            })
    ensemble_capital_ranked = _aggregate_ensemble_candidate_rows(
        ensemble_capital_rows, min_agree=1
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_vote_precedes_capital_adjusted_score_and_equal_votes_use_r2",
        [("C", 3), ("B", 2), ("A", 2)],
        [
            (row["ticker"], row["ensemble_vote_count"])
            for row in ensemble_capital_ranked
        ],
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_candidate_preserves_member_specific_original_quality_ranks",
        [f"m{idx}" for idx in range(6)],
        sorted(ensemble_ranked[0]["ensemble_member_quality_rank_by_key"]),
    )

    partial_ensemble_rows = [
        {
            "ticker": "A",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.10,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.90,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "A",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.10,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-20",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "B",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.20,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.50,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "B",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.20,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": 0.50,
                "available": True,
                "unavailable_reason": "",
                "score_date": "2017-06-21",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "C",
            "ensemble_member_key": "m0",
            "params_obj": base,
            "sort_value": 0.05,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-22",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
        {
            "ticker": "C",
            "ensemble_member_key": "m1",
            "params_obj": base,
            "sort_value": 0.05,
            "proj_cost": 100.0,
            "use_breakout_quality_ranking": True,
            "breakout_quality_rank": {
                "score": None,
                "available": False,
                "unavailable_reason": "missing_ticker_date_score",
                "score_date": "2017-06-21",
                "score_source": "selection_point_in_time",
                "shared_group_score": True,
                "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            },
        },
    ]
    partial_ensemble_ranked = _aggregate_ensemble_candidate_rows(
        partial_ensemble_rows, min_agree=2
    )
    partial_by_ticker = {row["ticker"]: row for row in partial_ensemble_ranked}
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "partial_ensemble_score_availability_falls_back_without_excluding_candidate",
        (
            ["B", "C", "A"],
            False,
            "partial_ensemble_member_score_availability",
            "partial_available_fallback",
            1,
            1,
            ["m0", "m1"],
        ),
        (
            [row["ticker"] for row in partial_ensemble_ranked],
            partial_by_ticker["A"]["breakout_quality_rank"]["available"],
            partial_by_ticker["A"]["breakout_quality_rank"]["unavailable_reason"],
            partial_by_ticker["A"]["ensemble_quality_score_availability"],
            partial_by_ticker["A"]["ensemble_quality_score_available_member_count"],
            partial_by_ticker["A"]["ensemble_quality_score_unavailable_member_count"],
            sorted(partial_by_ticker["A"]["ensemble_member_quality_rank_by_key"]),
        ),
    )

    same_date_inconsistent_rows = [dict(row) for row in partial_ensemble_rows[:2]]
    same_date_inconsistent_rows[1] = dict(same_date_inconsistent_rows[1])
    same_date_inconsistent_rows[1]["breakout_quality_rank"] = dict(
        same_date_inconsistent_rows[1]["breakout_quality_rank"],
        score_date="2017-06-22",
    )
    try:
        _aggregate_ensemble_candidate_rows(same_date_inconsistent_rows, min_agree=2)
        same_date_inconsistency_rejected = False
    except ValueError as exc:
        same_date_inconsistency_rejected = "ticker／score_date" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "same_ticker_same_score_date_availability_mismatch_remains_fail_fast",
        True,
        same_date_inconsistency_rejected,
    )

    reentry_params = replace(
        base,
        use_breakout_reclaim_reentry=True,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    original_rank = {
        "score": 0.731,
        "available": True,
        "unavailable_reason": "",
        "score_date": "2021-12-30",
        "shared_group_score": True,
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    }
    reentry_position = {
        "ticker": "3706",
        "entry_type": "normal",
        "entry_fill_price": 100.0,
        "pure_buy_price": 100.0,
        "initial_stop": 90.0,
        "sl": 90.0,
        "qty": 10,
        "initial_qty": 10,
        "breakout_quality_score": original_rank["score"],
        "breakout_quality_score_date": original_rank["score_date"],
        "breakout_quality_rank": dict(original_rank),
        "use_breakout_quality_ranking": True,
    }
    reentry_watch = create_breakout_reentry_watch_state(
        reentry_position,
        exit_date=pd.Timestamp("2022-01-05"),
        params=reentry_params,
        exit_atr=2.0,
        exit_qty=10,
    )
    reentry_trigger_date = pd.Timestamp("2022-01-17")
    reentry_signal = create_breakout_reentry_signal_state(
        reentry_watch,
        close_price=float(reentry_watch["confirm_price"]),
        atr=2.0,
        params=reentry_params,
        ticker="3706",
        signal_date=reentry_trigger_date,
    )
    inherited_rank = resolve_breakout_quality_rank(reentry_signal)
    from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_CANONICAL_RUNTIME
    from filters.breakout_quality.runtime import breakout_quality_ranking_source_context

    # This case validates the historical fixed-signal-score continuation contract.
    # Isolate it from the promoted daily-universal workflow, whose intended contract
    # refreshes the score on each latest completed trading day.
    with breakout_quality_ranking_source_context(
        score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ):
        with patch(
            "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
            side_effect=AssertionError("fixed-signal re-entry 不得以確認日重新查 score table"),
        ):
            resolved_reentry_rank = _resolve_candidate_quality_ranking(
                params=reentry_params,
                ticker="3706",
                signal_date=reentry_trigger_date,
                signal_state=reentry_signal,
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "reentry_keeps_trigger_date_but_inherits_original_breakout_score_date",
        ("2022-01-17", "2021-12-30", 0.731),
        (
            pd.Timestamp(reentry_signal["signal_date"]).strftime("%Y-%m-%d"),
            inherited_rank["score_date"],
            inherited_rank["score"],
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "reentry_ranking_uses_inherited_score_without_runtime_lookup_on_trigger_date",
        ("2021-12-30", 0.731),
        (resolved_reentry_rank["score_date"], resolved_reentry_rank["score"]),
    )

    incomplete_member_rank_position = {
        "ticker": "3706",
        "use_breakout_quality_ranking": True,
        "breakout_quality_rank": dict(original_rank),
        "_ensemble_member_params_by_key": {"m0": reentry_params, "m1": reentry_params},
        "_ensemble_member_quality_rank_by_key": {"m0": dict(original_rank)},
    }
    try:
        list(_iter_reentry_watch_targets(incomplete_member_rank_position, reentry_params, {}))
        missing_member_rank_rejected = False
    except ValueError as exc:
        missing_member_rank_rejected = "member=m1" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_reentry_rejects_partial_member_quality_rank_mapping",
        True,
        missing_member_rank_rejected,
    )

    ensemble_source = build_static_active_param_ensemble_payload(
        members=[
            {"member_index": 1, "seed": 101, "params": params_to_json_dict(base)},
            {"member_index": 2, "seed": 202, "params": params_to_json_dict(replace(base, high_len=205))},
        ],
        random_seed_ensemble={"enabled": True, "seed_count": 2, "min_agree": 2},
        selector="base_finalists_agree",
    )
    (
        ensemble_kind,
        ensemble_no_filter,
        ensemble_quality,
        _ensemble_no_payload,
        _ensemble_quality_payload,
        ensemble_policy,
    ) = _build_controlled_param_source_pair(
        {"kind": "static_active_param_ensemble", "payload": ensemble_source},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=0.01,
    )
    ensemble_switches = [
        (
            bool(left_member["params"]["use_breakout_quality_filter"]),
            bool(right_member["params"]["use_breakout_quality_filter"]),
            int(left_member["params"]["high_len"]),
            int(right_member["params"]["high_len"]),
            float(left_member["params"]["fixed_risk"]),
            float(right_member["params"]["fixed_risk"]),
        )
        for left_member, right_member in zip(
            ensemble_no_filter["params_ensemble"], ensemble_quality["params_ensemble"]
        )
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "static_trade_ensemble_preserves_members_and_only_toggles_filter",
        (
            "static_active_param_ensemble",
            [(False, True, 201, 201, 0.01, 0.01), (False, True, 205, 205, 0.01, 0.01)],
            (2, 2),
        ),
        (
            ensemble_kind,
            ensemble_switches,
            (ensemble_policy["seed_count"], ensemble_policy["min_agree"]),
        ),
    )

    broken_ensemble = json.loads(json.dumps(ensemble_quality))
    broken_ensemble["params_ensemble"][1]["params"]["high_len"] += 1
    try:
        _assert_controlled_ensemble_pair(ensemble_no_filter, broken_ensemble)
        ensemble_extra_difference_rejected = False
    except ValueError as exc:
        ensemble_extra_difference_rejected = "high_len" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "ensemble_additional_member_difference_is_rejected",
        True,
        ensemble_extra_difference_rejected,
    )

    rolling_single_payload = {
        "schema_type": "rolling_oos_param_set",
        "schema_version": 1,
        "usage": "validation_only",
        "summary": {"oos_end_date": "2022-12-31"},
        "params_by_effective_date": {
            "2021-01-01": params_to_json_dict(base),
            "2022-01-01": params_to_json_dict(replace(base, high_len=205)),
        },
        "params_by_oos_year": {
            "2021": params_to_json_dict(base),
            "2022": params_to_json_dict(replace(base, high_len=205)),
        },
        "folds": [
            {"effective_start": "2021-01-01", "effective_end": "2021-12-31"},
            {"effective_start": "2022-01-01", "effective_end": "2022-12-31"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        rolling_single_path = Path(tmp_dir) / "roos_base_best.json"
        rolling_single_path.write_text(json.dumps(rolling_single_payload), encoding="utf-8")
        rolling_single_source = _load_param_source(rolling_single_path)
        rolling_single_pair = _build_controlled_param_source_pair(
            rolling_single_source,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
        )
    rolling_single_left = rolling_single_pair[1]["params_by_effective_date"]["2022-01-01"]
    rolling_single_right = rolling_single_pair[2]["params_by_effective_date"]["2022-01-01"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_oos_single_param_schedule_is_supported_without_changing_effective_dates",
        ("rolling_oos_param_schedule", False, True, 205, 205),
        (
            rolling_single_pair[0],
            rolling_single_left["use_breakout_quality_filter"],
            rolling_single_right["use_breakout_quality_filter"],
            rolling_single_left["high_len"],
            rolling_single_right["high_len"],
        ),
    )

    rolling_payload = {
        "schema_type": "optimizer_active_param_ensemble",
        "schema_version": 1,
        "mode": "rolling",
        "usage": "validation_only",
        "type": "outer_rolling_oos_param_set",
        "random_seed_ensemble": {"enabled": True, "seed_count": 2, "min_agree": 2},
        "summary": {"oos_end_date": "2022-12-31"},
        "params_by_effective_date": {"2021-01-01": params_to_json_dict(base), "2022-01-01": params_to_json_dict(base)},
        "params_by_oos_year": {"2021": params_to_json_dict(base), "2022": params_to_json_dict(base)},
        "params_ensemble_by_effective_date": {
            "2021-01-01": [
                {"member_index": 1, "params": params_to_json_dict(base)},
                {"member_index": 2, "params": params_to_json_dict(replace(base, high_len=205))},
            ],
            "2022-01-01": [
                {"member_index": 1, "params": params_to_json_dict(base)},
                {"member_index": 2, "params": params_to_json_dict(replace(base, high_len=205))},
            ],
        },
        "folds": [
            {"effective_start": "2021-01-01", "effective_end": "2021-12-31"},
            {"effective_start": "2022-01-01", "effective_end": "2022-12-31"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        rolling_path = Path(tmp_dir) / "roos_base_finalists_agree.json"
        rolling_path.write_text(json.dumps(rolling_payload), encoding="utf-8")
        rolling_source = _load_param_source(rolling_path)
        rolling_pair = _build_controlled_param_source_pair(
            rolling_source,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
        )
    rolling_all_off_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    rolling_all_off_left = rolling_all_off_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_all_off_right = rolling_all_off_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_optional_entry_filter_gate_keeps_schedule_and_forces_all_five_filters_off",
        (
            "rolling_active_param_ensemble",
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            2,
        ),
        (
            rolling_all_off_pair[0],
            tuple(
                rolling_all_off_left[field]
                for field in OPTIONAL_ENTRY_FILTER_FIELDS
            ),
            tuple(
                rolling_all_off_right[field]
                for field in OPTIONAL_ENTRY_FILTER_FIELDS
            ),
            (
                rolling_all_off_left["use_breakout_quality_ranking"],
                rolling_all_off_right["use_breakout_quality_ranking"],
            ),
            len(rolling_all_off_pair[1]["params_ensemble_by_effective_date"]),
        ),
    )
    rolling_dl_filter_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    rolling_dl_left = rolling_dl_filter_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_dl_right = rolling_dl_filter_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_binary_dl_replacement_gate_keeps_original_sort_and_forces_optional_filters_off",
        (
            "rolling_active_param_ensemble",
            (False, False, False, False, False),
            (False, False, False, False, False),
            (False, True),
            (False, False),
            2,
        ),
        (
            rolling_dl_filter_pair[0],
            tuple(rolling_dl_left[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            tuple(rolling_dl_right[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            (
                rolling_dl_left["use_breakout_quality_filter"],
                rolling_dl_right["use_breakout_quality_filter"],
            ),
            (
                rolling_dl_left["use_breakout_quality_ranking"],
                rolling_dl_right["use_breakout_quality_ranking"],
            ),
            len(rolling_dl_filter_pair[1]["params_ensemble_by_effective_date"]),
        ),
    )

    rolling_rule_ablation_pair = _build_controlled_param_source_pair(
        rolling_source,
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_HARD_FILTER,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
        shared_param_overrides={
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": False,
            "use_kc": False,
        },
    )
    rolling_rule_left = rolling_rule_ablation_pair[1][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    rolling_rule_right = rolling_rule_ablation_pair[2][
        "params_ensemble_by_effective_date"
    ]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_binary_dl_rule_ablation_applies_same_history_reentry_kc_overrides_to_both_sides",
        (
            (False, False, False),
            (False, False, False),
            (False, True),
            (False, False),
            2,
        ),
        (
            tuple(
                rolling_rule_left[field]
                for field in (
                    "use_history_threshold",
                    "use_breakout_reclaim_reentry",
                    "use_kc",
                )
            ),
            tuple(
                rolling_rule_right[field]
                for field in (
                    "use_history_threshold",
                    "use_breakout_reclaim_reentry",
                    "use_kc",
                )
            ),
            (
                rolling_rule_left["use_breakout_quality_filter"],
                rolling_rule_right["use_breakout_quality_filter"],
            ),
            (
                rolling_rule_left["use_breakout_quality_ranking"],
                rolling_rule_right["use_breakout_quality_ranking"],
            ),
            len(
                rolling_rule_ablation_pair[1][
                    "params_ensemble_by_effective_date"
                ]
            ),
        ),
    )

    finalist_best_payload = json.loads(json.dumps(rolling_payload))
    finalist_best_payload["selector"] = "base_finalist_best"
    finalist_best_payload["random_seed_ensemble"] = {
        "enabled": False, "seed_count": 1, "min_agree": 1, "policy_name": "base_finalist_best"
    }
    for effective_date in list(finalist_best_payload["params_ensemble_by_effective_date"]):
        member = finalist_best_payload["params_ensemble_by_effective_date"][effective_date][0]
        member["policy"] = "base_finalist_best"
        finalist_best_payload["params_ensemble_by_effective_date"][effective_date] = [member]
    finalist_best_source = {"kind": "rolling_active_param_ensemble", "payload": finalist_best_payload}
    finalist_best_contract = _validate_requested_param_policy(
        finalist_best_source, PARAM_POLICY_BASE_FINALIST_BEST
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "finalist_best_score_ranking_policy_requires_single_runtime_member",
        ("base_finalist_best", 1, 1, 1),
        (
            finalist_best_contract["selector"],
            finalist_best_contract["member_count_min"],
            finalist_best_contract["member_count_max"],
            finalist_best_contract["min_agree"],
        ),
    )

    try:
        _validate_requested_param_policy(
            {"kind": "rolling_active_param_ensemble", "payload": rolling_payload},
            PARAM_POLICY_BASE_FINALIST_BEST,
        )
        wrong_selector_rejected = False
    except ValueError as exc:
        wrong_selector_rejected = "selector" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "finalist_best_policy_rejects_finalists_agree_source",
        True,
        wrong_selector_rejected,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        (root_path / "models").mkdir()
        resolved_best = _resolve_params_path(
            root=root_path, params_path=None,
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST, allow_static_diagnostic=False,
        )
        resolved_agree = _resolve_params_path(
            root=root_path, params_path=None,
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE, allow_static_diagnostic=False,
        )
    from core.strategy_param_artifacts import resolve_strategy_param_artifact_path

    expected_best = resolve_strategy_param_artifact_path(
        root_path, family="full", evaluation_mode="rolling", policy="base_finalist_best"
    )
    expected_agree = resolve_strategy_param_artifact_path(
        root_path, family="full", evaluation_mode="rolling", policy="base_finalists_agree"
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_param_policy_resolves_canonical_ssot_filenames",
        (expected_best.name, expected_agree.name),
        (resolved_best.name, resolved_agree.name),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "score_ranking_param_policies_use_isolated_output_directories",
        (
            "strategy_compare_score_ranking_base_finalist_best",
            "strategy_compare_score_ranking_base_finalists_agree",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
            ),
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "capital_aware_ranking_policies_use_isolated_output_directories",
        (
            "strategy_compare_score_ranking_base_finalist_best_capital_adjusted_score",
            "strategy_compare_score_ranking_base_finalist_best_capital_bucket_then_score",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_SCORE_RANKING,
                _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
                ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET,
            ),
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_param_policies_use_isolated_output_directories",
        (
            "strategy_compare_base_finalist_best",
            "strategy_compare_base_finalists_agree",
        ),
        (
            _comparison_output_dir_name(
                COMPARISON_MODE_HARD_FILTER,
                _comparison_labels(COMPARISON_MODE_HARD_FILTER),
                param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            ),
            _comparison_output_dir_name(
                COMPARISON_MODE_HARD_FILTER,
                _comparison_labels(COMPARISON_MODE_HARD_FILTER),
                param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
            ),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        compare_root = Path(tmp_dir)
        expected_compare_dir = compare_root / "strategy_compare_base_finalist_best"
        expected_compare_dir.mkdir()
        discovered_compare_dir = _first_existing_comparison_dir(
            compare_root,
            comparison_mode=COMPARISON_MODE_HARD_FILTER,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "hard_filter_policy_specific_output_is_discoverable",
        expected_compare_dir.name,
        discovered_compare_dir.name,
    )

    rolling_left = rolling_pair[1]["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]
    rolling_right = rolling_pair[2]["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rolling_oos_active_param_ensemble_is_supported_without_changing_schedule",
        ("rolling_active_param_ensemble", False, True, 2, 2),
        (
            rolling_pair[0],
            rolling_left["use_breakout_quality_filter"],
            rolling_right["use_breakout_quality_filter"],
            rolling_pair[5]["seed_count"],
            rolling_pair[5]["min_agree"],
        ),
    )

    capacity = _capacity_summary({
        "portfolio_capacity_rows": [
            {
                "Orderable_Candidates": 3,
                "Candidate_Supply_Gap": 7,
                "End_Position_Gap": 8,
                "Post_Execution_Positions": 2,
            },
            {
                "Orderable_Candidates": 0,
                "Candidate_Supply_Gap": 8,
                "End_Position_Gap": 8,
                "Post_Execution_Positions": 2,
            },
        ]
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_candidate_and_position_gap_summary",
        (2, 1.5, 1, 2, 15, 2, 16, 2.0, 0),
        (
            capacity["sim_day_count"],
            capacity["avg_orderable_candidates"],
            capacity["zero_orderable_candidate_days"],
            capacity["candidate_supply_gap_days"],
            capacity["candidate_supply_gap_slot_days"],
            capacity["underfilled_end_days"],
            capacity["end_position_gap_slot_days"],
            capacity["avg_end_positions"],
            capacity["full_position_days"],
        ),
    )

    normalized_years = _normalize_yearly_completeness(pd.DataFrame([
        {
            "year": 2025,
            "is_full_year": True,
            "start_date": "2025-01-02",
            "end_date": "2025-12-31",
        },
        {
            "year": 2026,
            "is_full_year": True,
            "start_date": "2026-01-02",
            "end_date": "2026-03-02",
        },
    ]))
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_partial_final_year_is_not_marked_full",
        [True, False],
        list(normalized_years["is_full_year"]),
    )

    no_filter_history = pd.DataFrame([
        {"Date": "2025-01-03", "Ticker": "A", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-02", "候選日": "2025-01-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-01-10", "Ticker": "A", "Type": "全倉結算(指標)", "成交價": 13.0, "該筆總損益": 3000.0, "R_Multiple": 3.0},
        {"Date": "2025-02-03", "Ticker": "B", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-31", "候選日": "2025-02-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-02-10", "Ticker": "B", "Type": "全倉結算(停損)", "成交價": 9.0, "該筆總損益": -1000.0, "R_Multiple": -1.0},
        {"Date": "2025-03-03", "Ticker": "C", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-02-28", "候選日": "2025-03-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-03-10", "Ticker": "C", "Type": "期末強制結算", "成交價": 11.0, "該筆總損益": 1000.0, "R_Multiple": 1.0},
    ])
    quality_history = pd.DataFrame([
        {"Date": "2025-03-03", "Ticker": "C", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-02-28", "候選日": "2025-03-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-03-10", "Ticker": "C", "Type": "期末強制結算", "成交價": 11.0, "該筆總損益": 1000.0, "R_Multiple": 1.0},
        {"Date": "2025-04-03", "Ticker": "D", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-04-02", "候選日": "2025-04-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-04-10", "Ticker": "D", "Type": "期末強制結算", "成交價": 9.5, "該筆總損益": -500.0, "R_Multiple": -0.5},
    ])
    shared_scores = pd.DataFrame({
        "ticker": ["A", "B", "C", "D"],
        "date": ["2025-01-02", "2025-01-31", "2025-02-28", "2025-04-02"],
        SCORE_COLUMN: [0.40, 0.30, 0.80, 0.90],
    }).set_index(["ticker", "date"])[[SCORE_COLUMN]]
    attribution = build_trade_attribution(
        no_filter_trade_history=no_filter_history,
        quality_filter_trade_history=quality_history,
        shared_score_table=shared_scores,
        threshold=0.50,
        no_filter_portfolio_total_r=3.0,
        quality_filter_portfolio_total_r=0.5,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_partitions_common_and_exclusive_round_trips",
        (1, 2, 1, 3, 2),
        tuple(attribution["trade_partition"][key] for key in (
            "common_count", "no_filter_only_count", "quality_filter_only_count",
            "no_filter_total_count", "quality_filter_total_count",
        )),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_reconciles_portfolio_total_r",
        (3.0, 1.0, 0.5, -2.5, 0.0),
        (
            attribution["r_attribution"]["excluded_winner_r"],
            attribution["r_attribution"]["avoided_loser_r_abs"],
            attribution["r_attribution"]["replacement_loser_r_abs"],
            attribution["r_attribution"]["exclusive_selection_delta_r"],
            attribution["r_attribution"]["reconciliation_error_r"],
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_attribution_distinguishes_direct_filter_rejects",
        (2, 0, 0),
        (
            attribution["r_attribution"]["direct_filter_reject_count"],
            attribution["r_attribution"]["portfolio_path_displacement_count"],
            attribution["r_attribution"]["score_lookup_unavailable_count"],
        ),
    )

    native_payload = _to_json_native({
        "number": np.float64(1.25),
        "date": pd.Timestamp("2026-07-26"),
        "non_finite": np.float64(np.nan),
    })
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_json_payload_is_native_and_strict",
        {"number": 1.25, "date": "2026-07-26T00:00:00", "non_finite": None},
        native_payload,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "comparison_json_number_is_builtin_float",
        True,
        type(native_payload["number"]) is float,
    )

    summary["controlled_param_difference"] = ["use_breakout_quality_filter"]
    return results, summary

def validate_breakout_quality_binary_dl_param_adaptation_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_BINARY_DL_PARAM_ADAPTATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    base = V16StrategyParams()

    from filters.breakout_quality.binary_pit_score_store import (
        BINARY_PIT_SCORE_SOURCE,
        build_pass_condition_from_binary_point_in_time_scores,
        load_binary_point_in_time_score_table,
    )
    from filters.breakout_quality.runtime import (
        BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV,
        BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV,
        BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV,
        breakout_quality_filter_source_context,
        build_breakout_quality_filter_pass_condition,
        get_breakout_quality_filter_source_context,
    )
    from strategies.breakout.search_space import (
        BREAKOUT_OPTIMIZER_SEARCH_SPACE,
        build_trial_params,
    )
    from filters.breakout_quality.strategy_compare_engine import (
        _run_scenario as run_strategy_comparison_scenario,
        run_comparison as run_strategy_comparison,
    )
    from filters.breakout_quality.strategy_param_training import (
        ALL_RULE_FILTERS_OFF_OVERRIDES,
        MIN_ROOS_SEARCH_FIELDS,
        _parse_args as parse_dl_param_adapt_args,
        _reuse_existing_min_roos_params_if_compatible,
        _validate_binary_pit_optimizer_coverage,
        build_min_roos_fold_overrides,
        run_param_adaptation_gate,
    )
    from tools.optimizer.outer_rolling_oos import (
        FOLD_FIXED_STRATEGY_OVERRIDES_KEY,
        _validate_optimizer_runtime_context,
        resolve_optimizer_session_spec_for_fold,
    )

    from config.breakout_quality import get_breakout_quality_workflow_settings

    workflow_settings = get_breakout_quality_workflow_settings()
    param_adapt_args = parse_dl_param_adapt_args([])
    baseline_member_params = params_to_json_dict(base)
    baseline_member_params.update(
        {
            "high_len": 220,
            "atr_len": 19,
            "atr_buy_tol": 1.0,
            "atr_times_init": 4.4,
            "atr_times_trail": 3.7,
            "use_history_threshold": False,
            "use_breakout_reclaim_reentry": True,
            "use_kc": True,
        }
    )
    synthetic_baseline_contract = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2021-12-31",
            "train_window_months": 120,
            "oos_horizon_months": 12,
        },
        "payload": {
            "params_ensemble_by_effective_date": {
                "2021-01-01": [
                    {"member_index": 1, "params": baseline_member_params}
                ]
            }
        }
    }
    p2_overrides = build_min_roos_fold_overrides(
        baseline_contract=synthetic_baseline_contract,
        args=param_adapt_args,
        training_dl_enabled=False,
    )
    p3_overrides = build_min_roos_fold_overrides(
        baseline_contract=synthetic_baseline_contract,
        args=param_adapt_args,
        training_dl_enabled=True,
    )
    p2_fixed = p2_overrides["2021-01-01"]
    p3_fixed = p3_overrides["2021-01-01"]
    resolved_fold_spec = resolve_optimizer_session_spec_for_fold(
        {
            "fixed_strategy_param_overrides": {"fixed_risk": 0.01},
            FOLD_FIXED_STRATEGY_OVERRIDES_KEY: p3_overrides,
        },
        oos_start_date="2021-01-01",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_min_roos_fold_contract",
        (
            tuple(MIN_ROOS_SEARCH_FIELDS),
            False,
            True,
            True,
            True,
            True,
            workflow_settings.experiment_profile,
        ),
        (
            tuple(MIN_ROOS_SEARCH_FIELDS),
            p2_fixed["use_breakout_quality_filter"],
            p3_fixed["use_breakout_quality_filter"],
            all(not p2_fixed[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
            all(
                p3_fixed.get(key) == value
                for key, value in ALL_RULE_FILTERS_OFF_OVERRIDES.items()
            ),
            resolved_fold_spec["fixed_strategy_param_overrides"]
            ["use_breakout_quality_filter"],
            param_adapt_args.experiment_profile,
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        completed_params_path = Path(tmp) / "roos_base_best.json"
        completed_member_params = dict(p2_fixed)
        for field_name in MIN_ROOS_SEARCH_FIELDS:
            completed_member_params[field_name] = baseline_member_params[field_name]
        completed_params_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "first_oos_date": "2021-01-01",
                        "last_oos_date": "2021-12-01",
                        "train_window_months": 120,
                        "oos_horizon_months": 12,
                        "trials_per_fold": int(param_adapt_args.trials_per_fold),
                    },
                    "params_ensemble_by_effective_date": {
                        "2021-01-01": [
                            {"member_index": 1, "params": completed_member_params}
                        ]
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        reused_completed_payload = _reuse_existing_min_roos_params_if_compatible(
            prior_preflight={"runtime_identity_sha256": "synthetic-current"},
            contract={"runtime_identity_sha256": "synthetic-current"},
            params_path=completed_params_path,
            baseline_contract=synthetic_baseline_contract,
            fold_overrides=p2_overrides,
            args=param_adapt_args,
            training_dl_enabled=False,
            arm_id="P2_HISTORY",
        )
        stamped_completed_payload = json.loads(
            completed_params_path.read_text(encoding="utf-8")
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "completed_min_roos_month_bucket_artifact_is_reused_after_postrun_validation_failure",
        (True, "min_roos_training", "2021-12-01"),
        (
            reused_completed_payload is not None,
            dict(
                stamped_completed_payload.get("breakout_quality_param_adaptation")
                or {}
            ).get("mode"),
            dict(reused_completed_payload.get("meta") or {}).get("last_oos_date")
            if isinstance(reused_completed_payload, dict)
            else None,
        ),
    )

    class _RiskOnlySession:
        optimizer_fixed_tp_percent = 0.0

        def __init__(self, values):
            self.values = dict(values)

        def has_fixed_strategy_param(self, field_name):
            return field_name in self.values

        def get_fixed_strategy_param(self, field_name, default=None):
            return self.values.get(field_name, default)

        def resolve_optimizer_tp_percent(self, trial, *, fixed_tp_percent):
            return fixed_tp_percent

    class _RiskOnlyTrial:
        def __init__(self):
            self.calls = []

        def suggest_int(self, field_name, low, high, step=1):
            self.calls.append(field_name)
            return int(low)

        def suggest_float(self, field_name, low, high, step=None):
            self.calls.append(field_name)
            return float(low)

        def suggest_categorical(self, field_name, choices):
            self.calls.append(field_name)
            return list(choices)[0]

    risk_trial = _RiskOnlyTrial()
    risk_params = build_trial_params(_RiskOnlySession(p2_fixed), risk_trial)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_searches_only_min_roos_fields",
        (
            set(MIN_ROOS_SEARCH_FIELDS),
            int(BREAKOUT_OPTIMIZER_SEARCH_SPACE["high_len"]["low"]),
            False,
            False,
            False,
        ),
        (
            set(risk_trial.calls),
            risk_params.high_len,
            risk_params.use_bb,
            risk_params.use_kc,
            risk_params.use_breakout_quality_filter,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        manifest_path = tmp_root / "manifest.json"
        scores_path = tmp_root / "scores.csv"
        score_table = pd.DataFrame(
            {
                "ticker": ["2330", "2330"],
                "date": ["2020-01-02", "2020-01-03"],
                "group_index": [1, 2],
                "dl_quality_score": [0.70, 0.30],
                "fold_id": ["fold_1", "fold_1"],
                "model_information_cutoff": ["2020-01-01", "2020-01-01"],
            }
        )
        score_table.to_csv(scores_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_type": "binary_point_in_time_scores",
                    "score_table": {"row_count": 2},
                    "score_period": {
                        "start": "2020-01-02",
                        "end": "2020-01-03",
                    },
                }
            ),
            encoding="utf-8",
        )
        load_binary_point_in_time_score_table.cache_clear()
        indexed, _manifest = load_binary_point_in_time_score_table(
            str(manifest_path), str(scores_path)
        )
        frame = pd.DataFrame(
            {"close": [9.0, 10.0, 11.0]},
            index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]),
        )
        candidate = np.array([True, True, True])
        direct_pass = build_pass_condition_from_binary_point_in_time_scores(
            frame,
            ticker="2330",
            score_threshold=0.5,
            candidate_condition=candidate,
            manifest_path=str(manifest_path),
            scores_path=str(scores_path),
        )
        with breakout_quality_filter_source_context(
            score_source=BINARY_PIT_SCORE_SOURCE,
            manifest_path=str(manifest_path),
            scores_path=str(scores_path),
        ):
            runtime_pass = build_breakout_quality_filter_pass_condition(
                frame,
                ticker="2330",
                high_len=220,
                score_threshold=0.5,
                candidate_condition=candidate,
                project_root=str(tmp_root),
            )

        with patch(
            "filters.breakout_quality.strategy_compare_replay._run_scenario_inside_source_context",
            side_effect=lambda **_kwargs: get_breakout_quality_filter_source_context(),
        ):
            scenario_context = run_strategy_comparison_scenario(
                name="synthetic_binary_pit_replay",
                data_dir=tmp_root,
                param_source_kind="single_param",
                params=base,
                start_date="2020-01-02",
                end_date="2020-01-03",
                max_positions=10,
                enable_rotation=False,
                quiet=True,
                filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )

        class _ContextSession:
            def optimizer_runtime_context(self):
                return breakout_quality_filter_source_context(
                    score_source=BINARY_PIT_SCORE_SOURCE,
                    manifest_path=str(manifest_path),
                    scores_path=str(scores_path),
                )

        runtime_spec = {
            "runtime_context_spec": {
                "module": "filters.breakout_quality.runtime",
                "callable": "breakout_quality_filter_source_context",
                "kwargs": {
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            }
        }
        _validate_optimizer_runtime_context(_ContextSession(), runtime_spec)
        mismatch_rejected = False
        try:
            bad_spec = json.loads(json.dumps(runtime_spec))
            bad_spec["runtime_context_spec"]["kwargs"]["scores_path"] = str(
                tmp_root / "wrong.csv"
            )
            _validate_optimizer_runtime_context(_ContextSession(), bad_spec)
        except RuntimeError as exc:
            mismatch_rejected = "NON_RETRYABLE_RUNTIME_IDENTITY_ERROR" in str(exc)

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_store_and_optimizer_runtime_identity",
        (2, (True, True, False), (True, True, False), BINARY_PIT_SCORE_SOURCE, True),
        (
            len(indexed),
            tuple(direct_pass),
            tuple(runtime_pass),
            scenario_context.score_source,
            mismatch_rejected,
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        params_path = tmp_root / "params.json"
        params_path.write_text("{}", encoding="utf-8")
        manifest_path = tmp_root / "manifest.json"
        scores_path = tmp_root / "scores.csv"
        pd.DataFrame(
            {
                "ticker": ["2330"],
                "date": ["2021-01-04"],
                "group_index": [1],
                "dl_quality_score": [0.7],
                "fold_id": ["fold_2021"],
                "model_information_cutoff": ["2020-12-31"],
            }
        ).to_csv(scores_path, index=False, encoding="utf-8-sig")
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_type": "binary_point_in_time_scores",
                    "model_architecture": "inception_time_v1",
                    "experiment_profile": "unique_group_sampling",
                    "threshold": 0.5,
                    "score_period": {"start": "2021-01-04", "end": "2021-01-04"},
                    "score_table": {"row_count": 1},
                    "information_contract": "synthetic",
                }
            ),
            encoding="utf-8",
        )
        base_summary = {
            "total_return_pct": 10.0,
            "max_drawdown_pct": 5.0,
            "return_over_max_drawdown": 2.0,
            "annual_return_pct": 2.0,
            "log_r_squared": 0.9,
            "monthly_win_rate_pct": 60.0,
            "trade_count": 2,
            "win_rate_pct": 50.0,
            "payoff_ratio": 2.0,
            "expected_value_r": 0.5,
            "final_equity": 1_100_000.0,
            "avg_exposure_pct": 80.0,
            "max_exposure_pct": 100.0,
            "missed_buy_count": 0,
            "missed_sell_count": 0,
            "reserved_buy_fill_rate_pct": 100.0,
            "normal_trade_count": 2,
            "extended_trade_count": 0,
            "annual_trade_count": 2,
            "benchmark_return_pct": 1.0,
            "benchmark_max_drawdown_pct": 1.0,
            "benchmark_annual_return_pct": 1.0,
        }
        quality_summary = {**base_summary, "total_return_pct": 11.0, "trade_count": 1}
        scenario_calls = []

        def _fake_strategy_scenario(**kwargs):
            scenario_calls.append(kwargs)
            return {
                "marker": kwargs["name"],
                "equity_curve": pd.DataFrame(),
                "trade_history": pd.DataFrame(),
                "profile": {"portfolio_capacity_rows": [], "closed_trade_rows": []},
            }

        with (
            patch(
                "filters.breakout_quality.strategy_compare_engine._resolve_params_path",
                return_value=params_path,
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._load_param_source",
                return_value={"kind": "rolling_active_param_ensemble", "payload": {}},
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._validate_requested_param_policy",
                return_value={
                    "selector": "base_finalist_best",
                    "member_count_min": 1,
                    "member_count_max": 1,
                    "min_agree": 1,
                },
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._build_controlled_param_source_pair",
                return_value=(
                    "rolling_active_param_ensemble",
                    {},
                    {},
                    {},
                    {},
                    {"policy": "synthetic"},
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine.get_active_param_ensemble_date_range",
                return_value=("2021-01-01", "2021-12-31"),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._run_scenario",
                side_effect=_fake_strategy_scenario,
            ),
            patch("filters.breakout_quality.strategy_compare_engine._assert_shared_benchmark"),
            patch(
                "filters.breakout_quality.strategy_compare_engine._scenario_summary",
                side_effect=lambda payload: (
                    base_summary if payload["marker"] == "no_filter" else quality_summary
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._build_yearly_comparison",
                return_value=pd.DataFrame(
                    [
                        {
                            "year": 2021,
                            "no_filter_return_pct": 10.0,
                            "quality_filter_return_pct": 11.0,
                "score_ranking_return_pct": 12.0,
                            "delta_pct": 1.0,
                            "is_full_year": True,
                        }
                    ]
                ),
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine.write_trade_attribution_outputs"
            ),
            patch(
                "filters.breakout_quality.strategy_compare_engine._render_strategy_console_report",
                return_value="synthetic report",
            ),
            patch("filters.breakout_quality.strategy_compare_engine.print_artifact_paths"),
            redirect_stdout(io.StringIO()),
        ):
            direct_comparison = run_strategy_comparison(
                project_root=tmp_root,
                dataset="full",
                params_path=str(params_path),
                param_policy="base-finalist-best",
                comparison_mode="hard-filter",
                filter_id="breakout_quality_v1",
                model_architecture="inception_time_v1",
                experiment_profile="unique_group_sampling",
                output_dir_override=tmp_root / "out",
                comparison_start_date="2021-01-04",
                comparison_end_date="2021-01-04",
                quiet=True,
                hard_filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_strategy_compare_uses_pit_source_and_trade_count",
        (
            BINARY_PIT_SCORE_SOURCE,
            None,
            BINARY_PIT_SCORE_SOURCE,
            2,
            1,
        ),
        (
            direct_comparison["metadata"]["score_source"],
            scenario_calls[0].get("filter_source"),
            scenario_calls[1]["filter_source"]["score_source"],
            direct_comparison["no_filter"]["trade_count"],
            direct_comparison["quality_filter"]["trade_count"],
        ),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        manifest_path = temp_root / "manifest.json"
        scores_path = temp_root / "scores.csv"
        manifest_path.write_text("{}", encoding="utf-8")
        scores_path.write_text("synthetic", encoding="utf-8")
        env_keys = (
            BREAKOUT_QUALITY_FILTER_SCORE_SOURCE_ENV,
            BREAKOUT_QUALITY_BINARY_PIT_MANIFEST_ENV,
            BREAKOUT_QUALITY_BINARY_PIT_SCORES_ENV,
        )
        env_before = tuple(os.environ.get(key) for key in env_keys)
        project_root = Path(__file__).resolve().parents[2]

        def _probe_spawned_worker_environment(**_kwargs):
            probe_code = (
                "import json; "
                "from filters.breakout_quality.runtime import "
                "get_breakout_quality_filter_source_context; "
                "c=get_breakout_quality_filter_source_context(); "
                "print(json.dumps({'score_source': c.score_source, "
                "'manifest_path': c.manifest_path, 'scores_path': c.scores_path}))"
            )
            probe_env = dict(os.environ)
            existing_pythonpath = str(probe_env.get("PYTHONPATH") or "").strip()
            probe_env["PYTHONPATH"] = (
                str(project_root)
                if not existing_pythonpath
                else str(project_root) + os.pathsep + existing_pythonpath
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe_code],
                cwd=str(project_root),
                env=probe_env,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(completed.stdout.strip())

        with patch(
            "filters.breakout_quality.strategy_compare_replay._run_scenario_inside_source_context",
            side_effect=_probe_spawned_worker_environment,
        ):
            worker_probe = run_strategy_comparison_scenario(
                name="quality_filter",
                data_dir=temp_root,
                param_source_kind="rolling_active_param_ensemble",
                params={},
                start_date="2021-01-04",
                end_date="2021-01-04",
                max_positions=10,
                enable_rotation=False,
                quiet=True,
                filter_source={
                    "score_source": BINARY_PIT_SCORE_SOURCE,
                    "manifest_path": str(manifest_path),
                    "scores_path": str(scores_path),
                },
            )
        env_after = tuple(os.environ.get(key) for key in env_keys)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_strategy_compare_propagates_pit_source_to_spawned_workers",
        (
            BINARY_PIT_SCORE_SOURCE,
            str(manifest_path),
            str(scores_path),
            env_before,
        ),
        (
            worker_probe["score_source"],
            worker_probe["manifest_path"],
            worker_probe["scores_path"],
            env_after,
        ),
    )

    coverage_contract = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2026-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
        }
    }
    partial_pit = _validate_binary_pit_optimizer_coverage(
        binary_pit={
            "ready": True,
            "score_period": {"start": "2016-03-01", "end": "2026-03-02"},
        },
        baseline_contract=coverage_contract,
    )
    partial_coverage = dict(partial_pit["optimizer_coverage"])
    stale_tail_rejected = False
    try:
        _validate_binary_pit_optimizer_coverage(
            binary_pit={
                "ready": True,
                "score_period": {"start": "2016-03-01", "end": "2025-11-30"},
            },
            baseline_contract=coverage_contract,
        )
    except ValueError as exc:
        stale_tail_rejected = "尾端未覆蓋" in str(exc)
    no_overlap_rejected = False
    try:
        _validate_binary_pit_optimizer_coverage(
            binary_pit={
                "ready": True,
                "score_period": {"start": "2026-01-01", "end": "2026-03-02"},
            },
            baseline_contract=coverage_contract,
        )
    except ValueError as exc:
        no_overlap_rejected = "完全沒有重疊" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_partial_optimizer_history_uses_audited_dl_off_fallback",
        (
            {"start": "2011-01-01", "end": "2025-12-31"},
            "pass_through_dl_off",
            {
                "bootstrap_fallback_only": 0,
                "partial_score_history": 6,
                "full_score_history": 0,
            },
            True,
            True,
        ),
        (
            partial_pit["optimizer_required_period"],
            partial_coverage["pre_coverage_policy"],
            partial_coverage["coverage_mode_counts"],
            stale_tail_rejected,
            no_overlap_rejected,
        ),
    )

    from tools.filters.breakout_quality.build_binary_point_in_time_scores import (
        _build_binary_group_table,
        _train_fold as train_binary_pit_fold,
    )

    trade_path_group_events = pd.DataFrame(
        {
            "ticker": ["1101", "1101", "2330", "2330", "2603", "2603"],
            "date": [
                "2020-01-02", "2020-01-02",
                "2020-01-03", "2020-01-03",
                "2020-01-06", "2020-01-06",
            ],
            "group_index": [0, 0, 1, 1, 2, 2],
        }
    )
    trade_path_group_table = _build_binary_group_table(
        trade_path_group_events,
        np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
        np.array([-1, 1, 0, -1, -1, -1], dtype=np.int64),
    )
    mixed_eligible_rejected = False
    try:
        _build_binary_group_table(
            trade_path_group_events.iloc[:2].copy(),
            np.array([0, 0], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )
    except ValueError as exc:
        mixed_eligible_rejected = "混合eligible binary label" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_pit_trade_path_group_ignores_excluded_rows_but_rejects_eligible_conflicts",
        ([1, 2, 4], [1, 0, -1], True),
        (
            trade_path_group_table["event_row"].astype(int).tolist(),
            trade_path_group_table["label"].astype(int).tolist(),
            mixed_eligible_rejected,
        ),
    )

    pit_events = pd.DataFrame(
        {
            "ticker": ["1101", "1101", "1101"],
            "date": pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"]),
        }
    )
    pit_group_table = pd.DataFrame(
        {
            "event_row": [0, 1, 2],
            "ticker": ["1101", "1101", "1101"],
            "date": pd.to_datetime(["2018-01-02", "2019-01-02", "2020-01-02"]),
            "group_index": [0, 1, 2],
            "label_eval_end_date": pd.to_datetime(
                ["2018-02-28", "2019-12-31", "2020-02-28"]
            ),
        }
    )
    pit_bundle = SimpleNamespace(
        features=np.zeros((3, 2, 1), dtype=np.float32),
        context=np.zeros((3, 1), dtype=np.float32),
        labels=np.array([0, 1, 1], dtype=np.int64),
        events=pit_events,
        group_table=pit_group_table,
        profile=SimpleNamespace(
            training_sampling_mode="unique_ticker_date",
            augmentation_name="none",
            augmentation_parameters=lambda: {},
            lr_schedule_parameters=lambda: {},
            optimizer_name="adam",
            lr_schedule_name="none",
            training_weight_reduction="batch_weight_sum",
        ),
        model_spec=SimpleNamespace(as_manifest_payload=lambda: {"name": "synthetic"}),
    )
    pit_args = SimpleNamespace(
        evaluation_batch_size=4,
        evaluation_workers=0,
        epochs=2,
        batch_size=2,
        lr=0.001,
        weight_decay=0.0,
        gradient_clip_norm=1.0,
        seed=42,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        parallel_split_evaluation=False,
        train_prefetch_batches=0,
        experiment_profile="unique_group_sampling",
    )

    class _SyntheticModel:
        def eval(self):
            return self

        def state_dict(self):
            return {}

    class _SyntheticTorch:
        @staticmethod
        def save(payload, path):
            Path(path).write_text(json.dumps({"saved": True}), encoding="utf-8")

    pit_ids = {
        "train_ids": np.array([0], dtype=np.int64),
        "validation_ids": np.array([1], dtype=np.int64),
        "final_ids": np.array([0, 1], dtype=np.int64),
        "score_ids": np.array([2], dtype=np.int64),
    }
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._resolve_training_sampling_indices",
            side_effect=lambda events, labels, indices, **kwargs: (
                np.asarray(indices, dtype=np.int64),
                {"rows": len(indices)},
            ),
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._select_epoch_with_inner_validation",
            return_value={"best_epoch": 2, "best_validation_loss": 0.4},
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores.train_impl._fit_full_selection",
            return_value={"model": _SyntheticModel()},
        ),
        patch(
            "tools.filters.breakout_quality.build_binary_point_in_time_scores._predict_scores",
            return_value=np.array([0.75], dtype=np.float32),
        ),
    ):
        pit_fold_result = train_binary_pit_fold(
            _SyntheticTorch(),
            pit_bundle,
            {
                "fold_id": "fold_20200102_20201231",
                "score_start": pd.Timestamp("2020-01-02"),
                "score_end": pd.Timestamp("2020-12-31"),
            },
            pit_ids,
            args=pit_args,
            execution_plan=SimpleNamespace(),
            fold_dir=Path(tmpdir),
        )
    pit_frame = pit_fold_result["frame"]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_pit_fold_cutoff_precedes_score_and_exports_identity",
        ("2019-12-31", "2020-01-02", 0.75, True),
        (
            pit_fold_result["model_information_cutoff"],
            pit_frame.iloc[0]["date"],
            round(float(pit_frame.iloc[0]["dl_quality_score"]), 2),
            pd.Timestamp(pit_fold_result["model_information_cutoff"])
            < pd.Timestamp(pit_frame.iloc[0]["date"]),
        ),
    )

    synthetic_adapted = {
        "meta": {
            "first_oos_date": "2021-01-01",
            "last_oos_date": "2021-01-01",
            "train_window_months": 120,
            "oos_horizon_months": 12,
            "trials_per_fold": 1,
        },
        "summary": {"folds": 1},
        "params_ensemble_by_effective_date": {
            "2021-01-01": [
                {
                    "member_index": 1,
                    "params": {
                        **p2_fixed,
                        "atr_len": 7,
                        "atr_buy_tol": 2.1,
                        "atr_times_init": 4.2,
                        "atr_times_trail": 3.8,
                    },
                }
            ]
        },
        "params_by_effective_date": {},
        "params_by_oos_year": {},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        baseline_path = tmp_root / "models" / "roos_base_best.json"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_payload = {
            **synthetic_adapted,
            "params_ensemble_by_effective_date": {
                "2021-01-01": [
                    {"member_index": 1, "params": baseline_member_params}
                ]
            },
        }
        baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")
        p2_path = tmp_root / "p2.json"
        p3_path = tmp_root / "p3.json"
        p2_path.write_text(json.dumps(synthetic_adapted), encoding="utf-8")
        p3_payload = json.loads(json.dumps(synthetic_adapted))
        p3_payload["params_ensemble_by_effective_date"]["2021-01-01"][0][
            "params"
        ]["use_breakout_quality_filter"] = True
        p3_path.write_text(json.dumps(p3_payload), encoding="utf-8")
        orchestration_contract = {
            "path": baseline_path,
            "payload": baseline_payload,
            "sha256": "synthetic-baseline",
            "meta": {
                "window_mode": "fixed",
                "first_oos_date": "2021-01-01",
                "last_oos_date": "2021-01-01",
                "train_window_months": 120,
                "oos_horizon_months": 12,
            },
            "summary": {"folds": 1},
        }
        comparison_calls = []
        optimizer_calls = []

        def _fake_comparison(**kwargs):
            comparison_calls.append(kwargs)
            index = len(comparison_calls)
            return {
                "no_filter": {
                    "total_return_pct": 100.0 + index,
                    "max_drawdown_pct": 10.0,
                    "return_over_max_drawdown": 10.0,
                    "expected_value_r": 0.5,
                    "avg_exposure_pct": 80.0,
                    "trade_count": 100,
                },
                "quality_filter": {
                    "total_return_pct": 102.0 + index,
                    "max_drawdown_pct": 11.0,
                    "return_over_max_drawdown": 9.0,
                    "expected_value_r": 0.4,
                    "avg_exposure_pct": 70.0,
                    "trade_count": 90,
                },
            }

        def _fake_optimizer_arm(**kwargs):
            optimizer_calls.append(kwargs)
            is_p3 = bool(kwargs["training_dl_enabled"])
            return {
                "params_path": p3_path if is_p3 else p2_path,
                "summary": {
                    "status": "COMPLETED",
                    "parameter_set": "P3" if is_p3 else "P2",
                },
                "contract": {},
            }

        binary_pit = {
            "status": "READY",
            "ready": True,
            "manifest_path": "pit/manifest.json",
            "scores_path": "pit/scores.csv",
            "manifest_absolute": str(tmp_root / "pit" / "manifest.json"),
            "scores_absolute": str(tmp_root / "pit" / "scores.csv"),
            "manifest_sha256": "manifest-sha",
            "scores_sha256": "scores-sha",
            "score_period": {"start": "2011-01-01", "end": "2025-12-31"},
            "error": "",
        }
        with (
            patch(
                "filters.breakout_quality.strategy_param_training.load_model_artifact_contract",
                return_value=SimpleNamespace(),
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._load_baseline_contract",
                return_value=orchestration_contract,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._ensure_binary_pit",
                return_value=binary_pit,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training.configure_optuna_logging",
            ),
            patch(
                "filters.breakout_quality.strategy_param_training._run_optimizer_arm",
                side_effect=_fake_optimizer_arm,
            ),
            patch(
                "filters.breakout_quality.strategy_param_training.run_comparison",
                side_effect=_fake_comparison,
            ),
            redirect_stdout(io.StringIO()),
        ):
            orchestration_result = run_param_adaptation_gate(
                project_root=tmp_root,
                argv=[
                    "--dataset",
                    "full",
                    "--param-policy",
                    "base-finalist-best",
                    "--trials-per-fold",
                    "1",
                    "--max-positions",
                    "10",
                    "--rotation",
                    "off",
                ],
            )
        report_path = (
            tmp_root
            / "models/research/breakout_quality/binary_dl_filter_param_adaptation/risk_only_rolling/strategy_dl_filter_param_adapt_gate.md"
        )
        report_text = report_path.read_text(encoding="utf-8")

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "binary_dl_four_by_two_gate_orchestration_and_report",
        (
            2,
            (False, True),
            4,
            ("current", "all-off", "all-off", "all-off"),
            (False, True, True, True),
            (BINARY_PIT_SCORE_SOURCE,) * 4,
            ("2021-01-01",) * 4,
            ("2025-12-31",) * 4,
            4,
            "FOUR_BY_TWO_COMPLETE",
            True,
            True,
        ),
        (
            len(optimizer_calls),
            tuple(call["training_dl_enabled"] for call in optimizer_calls),
            len(comparison_calls),
            tuple(call["optional_entry_filter_policy"] for call in comparison_calls),
            tuple(
                bool(call.get("shared_param_overrides"))
                for call in comparison_calls
            ),
            tuple(
                str((call.get("hard_filter_source") or {}).get("score_source"))
                for call in comparison_calls
            ),
            tuple(str(call.get("comparison_start_date")) for call in comparison_calls),
            tuple(str(call.get("comparison_end_date")) for call in comparison_calls),
            len(orchestration_result["matrix"]),
            orchestration_result["status"],
            all(token in report_text for token in ("A0", "B0", "A3", "B3", "B3−A2")),
            all(token in report_text for token in ("100", "90", "binary_point_in_time（八操作點一致；process workers已傳遞）")),
        ),
    )

    summary["workflow"] = "binary_dl_filter_four_parameters_by_two_states"
    summary["binary_pit"] = "REQUIRED_AND_VALIDATED"
    return results, summary

def validate_breakout_quality_trade_path_label_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TRADE_PATH_LABEL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from filters.breakout_quality.contract import (
        TRADE_PATH_FILTER_ID,
        TRADE_PATH_LABEL_OBJECTIVE,
        expected_label_policy_for_filter_id,
    )
    from filters.breakout_quality.paths import resolve_filter_artifact_paths
    from core.backtest_core import run_v16_backtest
    from filters.breakout_quality.trade_path_label import (
        TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE,
        TRADE_PATH_SELECTION_BASELINE_OOS_MONTHS,
        TRADE_PATH_LABEL_CONTRACT_VERSION,
        TRADE_PATH_LABEL_ID,
        TRADE_PATH_LABEL_REASON_STATUS,
        TRADE_PATH_LABEL_STATUS_EXCLUDED,
        TRADE_PATH_LABEL_STATUS_PASS,
        TRADE_PATH_LABEL_STATUS_REJECT,
        TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE,
        TRADE_PATH_REASON_REALIZED_NET_PROFIT,
        TRADE_PATH_REASON_UNFILLED_DATA_END,
        TRADE_PATH_RESEARCH_FILTER_ID,
        build_trade_path_excluded_event_update,
        simulate_realized_trade_path_label,
    )
    from filters.breakout_quality.strategy_param_training import (
        validate_selection_historical_baseline_period,
    )
    from tools.filters.breakout_quality.build_trade_path_labels import (
        _load_valid_ticker_shard,
        _ticker_shard_path,
        _write_ticker_shard,
    )


    expected_teacher_dates = []
    teacher_cursor = pd.Timestamp(
        TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE
    ).normalize()
    teacher_last = pd.Timestamp(
        TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE
    ).normalize()
    while teacher_cursor <= teacher_last:
        expected_teacher_dates.append(teacher_cursor.strftime("%Y-%m-%d"))
        teacher_cursor = (
            teacher_cursor
            + pd.DateOffset(months=TRADE_PATH_SELECTION_BASELINE_OOS_MONTHS)
        ).normalize()
    valid_teacher_payload = {
        "params_ensemble_by_effective_date": {
            effective_date: [{"params": {"atr_len": 5}}]
            for effective_date in expected_teacher_dates
        }
    }
    valid_teacher_meta = {
        "first_oos_date": TRADE_PATH_SELECTION_BASELINE_FIRST_OOS_DATE,
        "last_oos_date": pd.Timestamp(
            TRADE_PATH_SELECTION_BASELINE_LAST_OOS_DATE
        ).to_period("M").start_time.strftime("%Y-%m-%d"),
    }
    accepted_dates = validate_selection_historical_baseline_period(
        payload=valid_teacher_payload,
        meta=valid_teacher_meta,
    )
    incomplete_teacher_payload = {
        "params_ensemble_by_effective_date": {
            key: value
            for key, value in valid_teacher_payload[
                "params_ensemble_by_effective_date"
            ].items()
            if key != expected_teacher_dates[-1]
        }
    }
    try:
        validate_selection_historical_baseline_period(
            payload=incomplete_teacher_payload,
            meta=valid_teacher_meta,
        )
    except ValueError:
        incomplete_schedule_rejected = True
    else:
        incomplete_schedule_rejected = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_historical_teacher_accepts_canonical_oos_boundary_and_validates_effective_schedule",
        (
            expected_teacher_dates[0],
            expected_teacher_dates[-1],
            len(expected_teacher_dates),
            True,
        ),
        (
            accepted_dates[0],
            accepted_dates[-1],
            len(accepted_dates),
            incomplete_schedule_rejected,
        ),
    )

    policy = expected_label_policy_for_filter_id(TRADE_PATH_FILTER_ID)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_label_identity_and_policy_are_isolated_from_9a",
        (
            TRADE_PATH_FILTER_ID,
            TRADE_PATH_LABEL_OBJECTIVE,
            "pending",
            "reuse_original_event",
            TRADE_PATH_LABEL_CONTRACT_VERSION,
            "realized_net_r_gt_zero",
            "realized_net_r_le_zero",
            "formal_single_stock_forced_closeout",
            "same_explicit_single_stock_sizing_capital",
            "exclude_from_binary_training",
        ),
        (
            TRADE_PATH_RESEARCH_FILTER_ID,
            policy.get("label_objective"),
            policy.get("initial_miss_buy_status"),
            policy.get("continuation_event_identity"),
            policy.get("label_contract_version"),
            policy.get("filled_positive_rule"),
            policy.get("filled_nonpositive_rule"),
            policy.get("filled_data_end_rule"),
            policy.get("sizing_capital_rule"),
            policy.get("unfilled_terminal_rule"),
        ),
    )

    params = V16StrategyParams(
        use_breakout_ema_filter=False,
        use_bb=False,
        use_kc=False,
        use_vol=False,
        use_breakout_return_filter=False,
        use_breakout_false_filter=False,
        use_history_threshold=False,
        use_breakout_reclaim_reentry=False,
        use_breakout_quality_filter=False,
        use_breakout_quality_ranking=False,
        high_len=3,
        atr_len=3,
        atr_buy_tol=1.0,
        atr_times_init=2.0,
        atr_times_trail=2.0,
        tp_percent=0.0,
        initial_capital=1_000_000.0,
        fixed_risk=0.01,
        max_position_cap_pct=0.3,
    )
    dates = pd.date_range("2020-01-01", periods=7, freq="B")
    frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 103.0, 101.0, 103.0, 104.0],
            "High": [97.0, 100.0, 102.0, 103.5, 102.0, 105.0, 105.0],
            "Low": [94.0, 97.0, 99.0, 102.0, 100.0, 102.0, 103.0],
            "Close": [96.0, 99.0, 101.0, 103.0, 101.0, 104.0, 104.0],
            "Volume": [1000.0] * 7,
        },
        index=dates,
    )
    frame.attrs["ticker"] = "2330"
    atr = np.full(len(frame), 2.0, dtype=np.float64)
    buy_condition = np.array([False, False, True, False, False, False, False])
    sell_condition = np.array([False, False, False, False, True, False, False])
    buy_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan, np.nan, np.nan])
    result = simulate_realized_trade_path_label(
        frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(atr, buy_condition, sell_condition, buy_limits),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_initial_miss_remains_pending_then_continuation_fill_gets_one_terminal_label",
        (
            LABEL_PASS,
            TRADE_PATH_LABEL_STATUS_PASS,
            TRADE_PATH_REASON_REALIZED_NET_PROFIT,
            "CONTINUATION_FILL",
            1,
            True,
            "IND_SELL",
            True,
        ),
        (
            result.label,
            result.status,
            result.reason,
            result.fill_type,
            result.continuation_wait_bars,
            result.initial_missed_buy,
            result.exit_reason,
            bool(result.realized_net_r is not None and result.realized_net_r > 0.0),
        ),
    )

    _stats, single_trade_logs = run_v16_backtest(
        frame,
        params=params,
        return_logs=True,
        precomputed_signals=(atr, buy_condition, sell_condition, buy_limits),
        ticker="2330",
    )
    single_trade = single_trade_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_continuation_completed_path_matches_single_stock_entry_exit_and_accounting",
        (
            result.fill_date,
            result.entry_price,
            result.exit_date,
            result.exit_price,
            result.exit_reason,
            result.realized_net_pnl,
            result.realized_net_r,
        ),
        (
            pd.Timestamp(single_trade["entry_date"]).strftime("%Y-%m-%d"),
            float(single_trade["entry_price"]),
            pd.Timestamp(single_trade["exit_date"]).strftime("%Y-%m-%d"),
            float(single_trade["exit_price"]),
            str(single_trade["exit_reason"]),
            float(single_trade["pnl"]),
            float(single_trade["r_mult"]),
        ),
    )

    partial_tp_params = replace(params, tp_percent=0.5, atr_times_trail=1.0)
    partial_tp_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 98.0, 105.0, 105.0, 106.0],
            "High": [97.0, 100.0, 102.0, 109.0, 106.0, 106.0, 107.0],
            "Low": [94.0, 97.0, 99.0, 97.5, 104.5, 104.0, 105.0],
            "Close": [96.0, 99.0, 101.0, 99.0, 105.5, 105.0, 106.0],
            "Volume": [1000.0] * 7,
        },
        index=pd.date_range("2020-04-01", periods=7, freq="B"),
    )
    partial_tp_frame.attrs["ticker"] = "2330"
    partial_tp_atr = np.full(len(partial_tp_frame), 5.0, dtype=np.float64)
    partial_tp_buy = np.array([False, False, True, False, False, False, False])
    partial_tp_sell = np.array([False, False, False, False, True, False, False])
    partial_tp_limits = np.array(
        [np.nan, np.nan, 100.0, np.nan, np.nan, np.nan, np.nan]
    )
    partial_tp_result = simulate_realized_trade_path_label(
        partial_tp_frame,
        ticker="2330",
        signal_pos=2,
        params=partial_tp_params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(
            partial_tp_atr,
            partial_tp_buy,
            partial_tp_sell,
            partial_tp_limits,
        ),
    )
    _partial_tp_stats, partial_tp_logs = run_v16_backtest(
        partial_tp_frame,
        params=partial_tp_params,
        return_logs=True,
        precomputed_signals=(
            partial_tp_atr,
            partial_tp_buy,
            partial_tp_sell,
            partial_tp_limits,
        ),
        ticker="2330",
    )
    partial_tp_trade = partial_tp_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_partial_take_profit_then_indicator_exit_matches_single_stock",
        (
            "INITIAL_FILL",
            "IND_SELL",
            float(partial_tp_trade["entry_price"]),
            float(partial_tp_trade["exit_price"]),
            float(partial_tp_trade["pnl"]),
            float(partial_tp_trade["r_mult"]),
        ),
        (
            partial_tp_result.fill_type,
            partial_tp_result.exit_reason,
            partial_tp_result.entry_price,
            partial_tp_result.exit_price,
            partial_tp_result.realized_net_pnl,
            partial_tp_result.realized_net_r,
        ),
    )

    reject_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 100.0, 96.0],
            "High": [97.0, 100.0, 102.0, 101.0, 97.0],
            "Low": [94.0, 97.0, 99.0, 99.0, 95.0],
            "Close": [96.0, 99.0, 101.0, 100.0, 96.0],
            "Volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-02-03", periods=5, freq="B"),
    )
    reject_frame.attrs["ticker"] = "2330"
    reject_atr = np.full(len(reject_frame), 2.0, dtype=np.float64)
    reject_buy = np.array([False, False, True, False, False])
    reject_sell = np.zeros(len(reject_frame), dtype=bool)
    reject_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan])
    reject_result = simulate_realized_trade_path_label(
        reject_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(reject_atr, reject_buy, reject_sell, reject_limits),
    )
    _reject_stats, reject_logs = run_v16_backtest(
        reject_frame,
        params=params,
        return_logs=True,
        precomputed_signals=(reject_atr, reject_buy, reject_sell, reject_limits),
        ticker="2330",
    )
    reject_trade = reject_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_negative_completed_path_is_reject_and_matches_single_stock",
        (
            LABEL_REJECT,
            TRADE_PATH_LABEL_STATUS_REJECT,
            TRADE_PATH_REASON_REALIZED_NET_NONPOSITIVE,
            "STOP",
            float(reject_trade["pnl"]),
            float(reject_trade["r_mult"]),
        ),
        (
            reject_result.label,
            reject_result.status,
            reject_result.reason,
            reject_result.exit_reason,
            reject_result.realized_net_pnl,
            reject_result.realized_net_r,
        ),
    )

    data_end_frame = pd.DataFrame(
        {
            "Open": [95.0, 98.0, 100.0, 100.0, 103.0],
            "High": [97.0, 100.0, 102.0, 103.0, 105.0],
            "Low": [94.0, 97.0, 99.0, 99.0, 102.0],
            "Close": [96.0, 99.0, 101.0, 102.0, 104.0],
            "Volume": [1000.0] * 5,
        },
        index=pd.date_range("2020-03-02", periods=5, freq="B"),
    )
    data_end_frame.attrs["ticker"] = "2330"
    data_end_atr = np.full(len(data_end_frame), 2.0, dtype=np.float64)
    data_end_buy = np.array([False, False, True, False, False])
    data_end_sell = np.zeros(len(data_end_frame), dtype=bool)
    data_end_limits = np.array([np.nan, np.nan, 101.0, np.nan, np.nan])
    data_end_result = simulate_realized_trade_path_label(
        data_end_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
    )
    _data_end_stats, data_end_logs = run_v16_backtest(
        data_end_frame,
        params=params,
        return_logs=True,
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
        ticker="2330",
    )
    data_end_trade = data_end_logs[0]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_filled_at_data_end_uses_formal_single_stock_forced_closeout",
        (
            True,
            "FORCED_CLOSEOUT",
            float(data_end_trade["exit_price"]),
            float(data_end_trade["pnl"]),
            float(data_end_trade["r_mult"]),
        ),
        (
            data_end_result.forced_closeout,
            data_end_result.exit_reason,
            data_end_result.exit_price,
            data_end_result.realized_net_pnl,
            data_end_result.realized_net_r,
        ),
    )

    unfilled_frame = data_end_frame.copy()
    unfilled_frame[["Open", "High", "Low", "Close"]] = [
        [95.0, 97.0, 94.0, 96.0],
        [98.0, 100.0, 97.0, 99.0],
        [100.0, 102.0, 99.0, 101.0],
        [105.0, 106.0, 104.0, 105.0],
        [106.0, 107.0, 105.0, 106.0],
    ]
    unfilled_frame.attrs["ticker"] = "2330"
    unfilled_result = simulate_realized_trade_path_label(
        unfilled_frame,
        ticker="2330",
        signal_pos=2,
        params=params,
        teacher_effective_date="2020-01-01",
        precomputed_signals=(data_end_atr, data_end_buy, data_end_sell, data_end_limits),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_missed_buy_without_fill_is_excluded_not_reject",
        (
            LABEL_INVALID,
            TRADE_PATH_LABEL_STATUS_EXCLUDED,
            True,
            True,
            None,
        ),
        (
            unfilled_result.label,
            unfilled_result.status,
            unfilled_result.reason in {
                TRADE_PATH_REASON_UNFILLED_DATA_END,
                "unfilled_terminated",
            },
            unfilled_result.initial_missed_buy,
            unfilled_result.realized_net_r,
        ),
    )

    excluded_update = build_trade_path_excluded_event_update(
        "teacher_params_unavailable",
        end_date="2020-01-01",
        teacher_effective_date=None,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_reason_registry_covers_all_terminal_statuses_and_builder_exclusions",
        (11, {"PASS", "REJECT", "EXCLUDED"}, LABEL_INVALID, "EXCLUDED"),
        (
            len(TRADE_PATH_LABEL_REASON_STATUS),
            set(TRADE_PATH_LABEL_REASON_STATUS.values()),
            excluded_update["label"],
            excluded_update["label_status"],
        ),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        shard_dir = Path(tmpdir) / "shards"
        shard_path = _ticker_shard_path(shard_dir, "2330")
        _write_ticker_shard(
            shard_path,
            [
                {"_event_index": 2, "label": LABEL_PASS},
                {"_event_index": 1, "label": LABEL_REJECT},
            ],
        )
        valid_shard = _load_valid_ticker_shard(
            shard_path, expected_event_indices={1, 2}
        )
        invalid_shard = _load_valid_ticker_shard(
            shard_path, expected_event_indices={1, 2, 3}
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "trade_path_resume_reuses_only_complete_ticker_shards",
            ([1, 2], True),
            (
                []
                if valid_shard is None
                else valid_shard["_event_index"].astype(int).tolist(),
                invalid_shard is None,
            ),
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_paths = resolve_filter_artifact_paths(
            tmpdir,
            TRADE_PATH_RESEARCH_FILTER_ID,
            "inception_time_v1",
            "unique_group_sampling",
        )
        add_check(
            results,
            "synthetic_breakout_quality",
            case_id,
            "trade_path_artifact_paths_keep_filter_identity",
            (TRADE_PATH_RESEARCH_FILTER_ID, True),
            (
                artifact_paths.filter_id,
                TRADE_PATH_RESEARCH_FILTER_ID in str(artifact_paths.model_dir),
            ),
        )


    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "tools" / "filters" / "breakout_quality" / "application.py").read_text(encoding="utf-8")
    builder_source = (
        project_root / "tools" / "filters" / "breakout_quality" / "build_trade_path_labels.py"
    ).read_text(encoding="utf-8")
    binary_pit_source = (
        project_root / "services" / "breakout_quality" / "binary_point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_menu_completes_model_artifacts_and_strategy_comparison_stays_separate",
        True,
        _source_has_render_menu_item_call(
            app_source,
            index=1,
            label="建立新Label → 重新訓練 → 模型預測報表",
            default=True,
        )
        and _source_has_render_menu_item_call(
            app_source,
            index=2,
            label="使用既有模型 → 更新Scores → 模型預測報表",
        )
        and _source_has_render_menu_item_call(
            app_source,
            index=3,
            label="查看Label與事件生命週期摘要",
        )
        and all(
            token in app_source
            for token in (
                '"build-trade-path-labels"',
                "本流程不執行策略績效比較",
                "apps/research.py compare",
            )
        )
        and "strategy-trade-path-label-gate" not in app_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_builder_reuses_formal_lifecycle_and_never_overwrites_9a",
        True,
        all(
            token in builder_source
            for token in (
                "source_filter_id == target_filter_id",
                "TRADE_PATH_RESEARCH_FILTER_ID",
                "derived_feature_bank_trade_path_relabel",
                "initial_miss_buy_status",
                "formal_single_stock_forced_closeout",
                "same_explicit_single_stock_sizing_capital",
                "exclude_from_binary_training",
                "build_signal_cache",
                "simulate_realized_trade_path_label",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "trade_path_binary_pit_uses_filter_specific_label_policy",
        True,
        "expected_label_policy_for_filter_id(str(args.filter_id))" in binary_pit_source
        and "expected_policy=DEFAULT_LABEL_POLICY.as_manifest_payload()" not in binary_pit_source,
    )

    summary["label_id"] = TRADE_PATH_LABEL_ID
    summary["research_filter_id"] = TRADE_PATH_RESEARCH_FILTER_ID
    summary["menu_scope"] = "model_prediction_only"
    summary["strategy_scope"] = "formal_compare_separate"
    return results, summary

def validate_breakout_quality_legacy_research_cleanup_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_LEGACY_RESEARCH_CLEANUP"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    project_root = Path(__file__).resolve().parents[2]

    retired_paths = (
        "tools/filters/breakout_quality/strategy_adapt.py",
        "tools/filters/breakout_quality/strategy_filter_gate.py",
        "tools/filters/breakout_quality/strategy_dl_filter_gate.py",
        "tools/filters/breakout_quality/strategy_trade_path_label_gate.py",
        "tools/audit/breakout_quality/pass_realization_gap.py",
        "tools/audit/breakout_quality/selection_strategy_realization.py",
        "tools/audit/breakout_quality/candidate_counterfactual_execution.py",
        "tools/audit/breakout_quality/portfolio_selection_pressure.py",
        "tools/audit/breakout_quality/orderable_feasible_alignment.py",
        "tools/audit/breakout_quality/selector_stage_translation.py",
        "tools/audit/breakout_quality/direct_r_calibration.py",
        "tools/audit/portfolio/score_ranking_capture.py",
        "doc/result_tmp.md",
    )
    retired_absent = tuple(
        not (project_root / relative_path).exists()
        for relative_path in retired_paths
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "retired_research_cli_audit_and_temp_output_are_absent",
        tuple(True for _ in retired_paths),
        retired_absent,
    )

    current_paths = (
        "filters/breakout_quality/strategy_comparison.py",
        "filters/breakout_quality/strategy_param_training.py",
        "tools/filters/breakout_quality/build_trade_path_labels.py",
        "services/breakout_quality/train.py",
        "tools/audit/catalog.py",
        "tools/validate/transient_code_maintenance.py",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "current_strategy_model_and_generic_audit_framework_remain_available",
        tuple(True for _ in current_paths),
        tuple((project_root / relative_path).exists() for relative_path in current_paths),
    )

    compatibility_paths = (
        "tools/filters/breakout_quality/build_pretraining_dataset.py",
        "tools/filters/breakout_quality/pretrain.py",
        "filters/breakout_quality/models/ts2vec.py",
        "filters/breakout_quality/models/mantis_v2.py",
        "filters/breakout_quality/models/moment.py",
        "config/compatibility/strategy_compare_history.py",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "required_historical_reconstruction_compatibility_is_preserved",
        tuple(True for _ in compatibility_paths),
        tuple((project_root / relative_path).exists() for relative_path in compatibility_paths),
    )

    application_source = (
        project_root / "tools/filters/breakout_quality/application.py"
    ).read_text(encoding="utf-8")
    current_docs = "\n".join(
        (project_root / relative_path).read_text(encoding="utf-8")
        for relative_path in ("doc/CMD.md", "doc/ARCHITECTURE.md")
    )
    retired_command_tokens = (
        "python -m tools.filters.breakout_quality.strategy_adapt",
        "python -m tools.filters.breakout_quality.strategy_filter_gate",
        "python -m tools.filters.breakout_quality.strategy_dl_filter_gate",
        "python -m tools.filters.breakout_quality.strategy_trade_path_label_gate",
        "audit-pass-realization-gap",
        "audit-selection-strategy-realization",
        "audit-candidate-counterfactual",
        "audit-selection-pressure",
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "retired_research_cli_is_not_routable_or_advertised",
        (False, False),
        (
            any(token in application_source for token in retired_command_tokens),
            any(token in current_docs for token in retired_command_tokens),
        ),
    )

    from tools.validate.transient_code_maintenance import summarize_transient_code_maintenance

    maintenance = summarize_transient_code_maintenance(project_root)
    maintenance_candidate_count = maintenance.get("candidate_count")
    maintenance_count_valid = (
        type(maintenance_candidate_count) is int
        and maintenance_candidate_count >= 0
    )
    maintenance_expected_status = (
        "REVIEW"
        if maintenance_count_valid and maintenance_candidate_count > 0
        else "CLEAN"
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "transient_code_maintenance_scan_is_advisory_and_internally_consistent",
        (True, True, True, True),
        (
            maintenance_count_valid,
            maintenance.get("status") == maintenance_expected_status,
            maintenance.get("needs_slimming") == bool(maintenance_candidate_count)
            if maintenance_count_valid
            else False,
            maintenance.get("advisory_only") is True,
        ),
    )

    summary["retired_paths"] = list(retired_paths)
    summary["current_replacements"] = list(current_paths)
    summary["compatibility_preserved"] = list(compatibility_paths)
    summary["maintenance"] = maintenance
    return results, summary

def validate_breakout_quality_strategy_readable_report_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STRATEGY_READABLE_REPORT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    project_root = Path(__file__).resolve().parents[2]

    contracts = (
        (
            "strategy_compare_pair",
            project_root / "filters/breakout_quality/strategy_compare_reporting.py",
            "strategy_comparison.md",
            "render_strategy_pair_simple_report",
        ),
        (
            "strategy_compare_multi_arm",
            project_root / "filters/breakout_quality/strategy_comparison.py",
            "strategy_comparison.md",
            "render_strategy_aggregate_report",
        ),
        (
            "strategy_parameter_adaptation",
            project_root / "services/optimizer/strategy_param_training.py",
            "strategy_dl_filter_param_adapt_gate.md",
            "_render_report",
        ),
    )
    contract_rows = []
    for name, source_path, markdown_filename, renderer_token in contracts:
        source = source_path.read_text(encoding="utf-8")
        contract_rows.append(
            (
                name,
                markdown_filename in source,
                renderer_token in source,
                "core.console_report" in source,
            )
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_formal_strategy_results_have_persistent_markdown_simple_report",
        True,
        all(row[1] for row in contract_rows),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "all_formal_strategy_results_have_console_readable_renderer",
        True,
        all(row[2] and row[3] for row in contract_rows),
    )

    comparison_source = (
        project_root / "filters/breakout_quality/strategy_comparison.py"
    ).read_text(encoding="utf-8")
    reporting_source = (
        project_root / "filters/breakout_quality/strategy_compare_reporting.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_compare_run_and_reuse_share_canonical_pair_readable_report_renderer",
        True,
        "materialize_strategy_pair_readable_report" in comparison_source
        and "render_strategy_pair_simple_report" in comparison_source
        and "materialize_strategy_pair_readable_report" in reporting_source
        and 'pair_dir / "strategy_comparison.md"' in comparison_source,
    )

    diagnostics_source = (
        project_root / "filters/breakout_quality/strategy_compare_diagnostics.py"
    ).read_text(encoding="utf-8")
    report_metrics_source = (project_root / "core/report_metrics.py").read_text(encoding="utf-8")
    report_style_source = (project_root / "core/report_style.py").read_text(encoding="utf-8")
    pit_audit_source = (
        project_root / "services/breakout_quality/point_in_time_audit.py"
    ).read_text(encoding="utf-8")
    multi_seed_source = (
        project_root / "filters/breakout_quality/strategy_multi_seed_robustness.py"
    ).read_text(encoding="utf-8")
    model_report_source = (
        project_root / "tools/filters/breakout_quality/report.py"
    ).read_text(encoding="utf-8")
    strategy_dashboard_source = (project_root / "core/strategy_dashboard.py").read_text(encoding="utf-8")
    optimizer_callbacks_source = (project_root / "services/optimizer/callbacks.py").read_text(encoding="utf-8")
    outer_roos_source = (project_root / "services/optimizer/outer_rolling_oos.py").read_text(encoding="utf-8")
    render_report_source = comparison_source[
        comparison_source.index("def render_strategy_aggregate_report("):
        comparison_source.index("def _run_directory(")
    ]
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_compare_main_report_surfaces_core_r_conversion_and_compact_execution",
        True,
        '"strategy_diagnostics.md"' in comparison_source
        and "核心策略結果" in render_report_source
        and "R 預測／轉化" in render_report_source
        and "資金／執行" in render_report_source
        and "5. 執行摘要" in render_report_source
        and "render_strategy_run_execution_table" in render_report_source
        and "報表分工" not in render_report_source
        and "render_strategy_r_analysis_table" in render_report_source
        and "metrics=CORE_STRATEGY_RESULT_METRICS" in comparison_source
        and "_contrast_table(" not in render_report_source
        and "_resource_aware_table(" not in render_report_source
        and "_selector_timing_table(" not in render_report_source
        and "R_ANALYSIS_GROUPED_SECTIONS" in diagnostics_source
        and "R_ANALYSIS_MERGED_METRICS" in diagnostics_source
        and '("實際交易", R_ACTUAL_TRADE_METRICS)' in report_metrics_source
        and '("模型預測", R_MODEL_PREDICTION_METRICS)' in report_metrics_source
        and '("選股轉換", R_SELECTION_TRANSLATION_METRICS)' in report_metrics_source
        and 'top_headers = ["分群", ""]' in diagnostics_source
        and 'bottom_headers = ["編號", "比較對象"]' in diagnostics_source
        and "best_worst_signals" in diagnostics_source
        and "_render_metric_notes" not in diagnostics_source
        and 'lines = ["註解", "----"]' not in diagnostics_source
        and '"top_target_r": _finite(metrics.get("top_decile_target_mean"))' in diagnostics_source
        and '"bottom_target_r": _finite(metrics.get("bottom_decile_target_mean"))' in diagnostics_source
        and '"top_target_r": top' in diagnostics_source
        and '"bottom_target_r": bottom' in diagnostics_source
        and '"r_conversion_efficiency", "RCE"' in report_metrics_source
        and "paired_trade_r_conversion_diagnostic" in diagnostics_source
        and "backfill_pair_r_conversion_diagnostic" in comparison_source
        and "Target %ile" in report_metrics_source
        and "Top-K" in report_metrics_source
        and "Opp gap" in report_metrics_source
        and "Top-R" in report_metrics_source
        and "Bottom-R" in report_metrics_source
        and "DL選擇R" in report_metrics_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_compare_progress_and_final_report_share_high_level_wall_time_summary",
        True,
        "arm_execution_timing" in comparison_source
        and '"execution_summary": execution_summary' in comparison_source
        and 'f"[RUN] {on_arm.arm_id} {on_arm.name} "' in comparison_source
        and 'f"[DONE] {on_arm.arm_id} {on_arm.name} "' in comparison_source
        and 'elapsed={format_elapsed(' in comparison_source
        and 'total={format_elapsed(' in comparison_source
        and 'render_section("5. 執行摘要")' in comparison_source
        and "render_strategy_run_execution_table" in comparison_source
        and "_selector_timing_table(" not in render_report_source,
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "multi_seed_robustness_reuses_strategy_compare_canonical_report_renderers_and_keeps_seed_extensions_separate",
        True,
        "render_strategy_aggregate_report" in multi_seed_source
        and "common_strategy_report" in multi_seed_source
        and "RoMD完整統計" in multi_seed_source
        and "設定中的同seed contrasts" in multi_seed_source
        and "歷年報酬跨seed完整統計" in multi_seed_source
        and "_upgrade_derived_report_summary" in multi_seed_source
        and "report_refreshed_at_utc" in multi_seed_source,
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_reports_share_metric_registry_and_project_wide_color_semantics",
        True,
        "PORTFOLIO_RESULT_METRICS" in report_metrics_source
        and "CORE_STRATEGY_RESULT_METRICS" in report_metrics_source
        and "PAIR_MAIN_METRICS" in report_metrics_source
        and "from core.report_metrics import" in comparison_source
        and "_report_reference_arm_id" in comparison_source
        and "best_worst_signals" in comparison_source
        and "styled_signal" in comparison_source
        and 'target="markdown"' in comparison_source
        and 'target="console"' in comparison_source
        and "判讀基準" not in render_report_source
        and "from core.report_metrics import PAIR_MAIN_METRICS" in reporting_source
        and "from core.report_style import" in reporting_source
        and "from core.report_style import" in multi_seed_source
        and "from core.report_style import" in pit_audit_source
        and "from core.report_style import" in model_report_source
        and "from core.report_style import" in strategy_dashboard_source
        and "from core.report_style import" in optimizer_callbacks_source
        and "from core.report_style import" in outer_roos_source
        and "SIGNAL_POSITIVE" in report_style_source
        and "markdown_signal" in report_style_source
        and not any(marker in report_style_source for marker in ("🟢", "🔴", "🟡", "⚪")),
    )
    from filters.breakout_quality.strategy_compare_diagnostics import (
        paired_trade_r_conversion_diagnostic,
    )
    baseline_trades = pd.DataFrame([
        {"Date": "2020-01-02", "Ticker": "AAA", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-01-01", "成交價": 10.0},
        {"Date": "2020-01-03", "Ticker": "BBB", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-01-02", "成交價": 10.0},
        {"Date": "2020-01-10", "Ticker": "AAA", "Type": "全倉結算", "成交價": 11.0, "R_Multiple": 0.5, "該筆總損益": 100.0},
        {"Date": "2020-01-11", "Ticker": "BBB", "Type": "全倉結算", "成交價": 9.0, "R_Multiple": -1.0, "該筆總損益": -100.0},
    ])
    active_trades = pd.DataFrame([
        {"Date": "2020-01-02", "Ticker": "AAA", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-01-01", "成交價": 10.0},
        {"Date": "2020-01-04", "Ticker": "CCC", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-01-03", "成交價": 10.0},
        {"Date": "2020-01-10", "Ticker": "AAA", "Type": "全倉結算", "成交價": 11.0, "R_Multiple": 0.5, "該筆總損益": 100.0},
        {"Date": "2020-01-12", "Ticker": "CCC", "Type": "全倉結算", "成交價": 12.0, "R_Multiple": 1.0, "該筆總損益": 100.0},
    ])
    baseline_targets = pd.DataFrame([
        {"ticker": "AAA", "trade_date": "2020-01-02", "signal_date": "2020-01-01", "target_raw_r": 1.0, "target_available": True},
        {"ticker": "BBB", "trade_date": "2020-01-03", "signal_date": "2020-01-02", "target_raw_r": 0.5, "target_available": True},
    ])
    active_targets = pd.DataFrame([
        {"ticker": "AAA", "trade_date": "2020-01-02", "signal_date": "2020-01-01", "target_raw_r": 1.0, "target_available": True},
        {"ticker": "CCC", "trade_date": "2020-01-04", "signal_date": "2020-01-03", "target_raw_r": 1.5, "target_available": True},
    ])
    rce_diag = paired_trade_r_conversion_diagnostic(
        baseline_trades, active_trades, baseline_targets, active_targets
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rce_uses_same_exclusive_completed_trade_universe_for_target_and_realized_edges",
        True,
        rce_diag.get("comparison_basis") == "target_covered_exclusive_realized_trade_mean_r"
        and abs(float(rce_diag.get("paired_target_selection_edge_r")) - 1.0) < 1e-12
        and abs(float(rce_diag.get("paired_realized_selection_edge_r")) - 2.0) < 1e-12
        and abs(float(rce_diag.get("exclusive_selection_delta_r")) - 2.0) < 1e-12
        and abs(float(rce_diag.get("r_conversion_efficiency")) - 2.0) < 1e-12,
    )

    partial_baseline_targets = baseline_targets.copy()
    partial_active_targets = pd.concat([
        active_targets,
        pd.DataFrame([{
            "ticker": "DDD", "trade_date": "2020-01-05", "signal_date": "2020-01-04",
            "target_raw_r": float("nan"), "target_available": False,
        }]),
    ], ignore_index=True)
    partial_active_trades = pd.concat([
        active_trades,
        pd.DataFrame([
            {"Date": "2020-01-05", "Ticker": "DDD", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-01-04", "成交價": 10.0},
            {"Date": "2020-01-13", "Ticker": "DDD", "Type": "全倉結算", "成交價": 11.0, "R_Multiple": 5.0, "該筆總損益": 100.0},
        ])
    ], ignore_index=True)
    partial_rce = paired_trade_r_conversion_diagnostic(
        baseline_trades, partial_active_trades, partial_baseline_targets, partial_active_targets
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "rce_excludes_missing_target_trade_from_both_edges_without_100pct_coverage_gate",
        True,
        partial_rce.get("complete_target_coverage") is False
        and int(partial_rce.get("active_only_count") or 0) == 2
        and int(partial_rce.get("active_only_target_covered_count") or 0) == 1
        and abs(float(partial_rce.get("target_coverage_rate")) - (2.0 / 3.0)) < 1e-12
        and abs(float(partial_rce.get("paired_target_selection_edge_r")) - 1.0) < 1e-12
        and abs(float(partial_rce.get("paired_realized_selection_edge_r")) - 2.0) < 1e-12
        and abs(float(partial_rce.get("r_conversion_efficiency")) - 2.0) < 1e-12,
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "strategy_diagnostics_reuse_existing_canonical_artifacts_without_raw_recalculation",
        True,
        '"raw_market_or_trade_recalculation": False' in diagnostics_source
        and '"audit"' in diagnostics_source
        and '"report"' in diagnostics_source
        and "selection_diagnostics" in diagnostics_source,
    )

    summary["strategy_output_contracts"] = [row[0] for row in contract_rows]
    return results, summary
