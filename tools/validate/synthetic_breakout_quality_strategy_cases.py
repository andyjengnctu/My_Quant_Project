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
    bind_checks,
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
    """Cross-module strategy-compare integration contract.

    Detailed menu, replay, PIT, robustness and report behavior lives in the
    dedicated validators.  This case keeps only the invariants that must hold
    across those modules: controlled pairs, ranking order, parameter schedule
    preservation, attribution reconciliation and strict JSON output.
    """
    case_id = "BREAKOUT_QUALITY_STRATEGY_COMPARISON"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    base = V16StrategyParams()
    common = {
        "breakout_quality_filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        "breakout_quality_score_threshold": float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
    }
    no_filter = replace(base, use_breakout_quality_filter=False, **common)
    quality_filter = replace(base, use_breakout_quality_filter=True, **common)
    _assert_controlled_param_pair(no_filter, quality_filter)
    check_true("controlled_pair_only_toggles_quality_filter", True)
    try:
        _assert_controlled_param_pair(no_filter, replace(quality_filter, high_len=int(base.high_len) + 1))
        extra_difference_rejected = False
    except ValueError as exc:
        extra_difference_rejected = "high_len" in str(exc)
    check_true("additional_param_difference_is_rejected", extra_difference_rejected)

    period = SimpleNamespace(
        execution_start=pd.Timestamp("2025-01-03").date(),
        required_signal_start=pd.Timestamp("2025-01-02").date(),
        available_from=pd.Timestamp("2025-01-02").date(),
        available_through=pd.Timestamp("2025-12-31").date(),
    )
    check(
        "strategy_comparison_starts_at_execution_window_not_signal_score_anchor",
        ("2025-01-03", "2025-12-31"),
        _resolve_comparison_period(period),
    )

    ranking_pair = _build_controlled_param_source_pair(
        {"kind": "single_param", "params": base},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=None,
        comparison_mode=COMPARISON_MODE_SCORE_RANKING,
        optional_entry_filter_policy=OPTIONAL_ENTRY_FILTER_POLICY_ALL_OFF,
    )
    ranking_left = params_to_json_dict(ranking_pair[1])
    ranking_right = params_to_json_dict(ranking_pair[2])
    check(
        "score_ranking_pair_isolates_ranking_and_optional_filters",
        ((False, False), (False, True), (False,) * len(OPTIONAL_ENTRY_FILTER_FIELDS)),
        (
            (ranking_left["use_breakout_quality_filter"], ranking_right["use_breakout_quality_filter"]),
            (ranking_left["use_breakout_quality_ranking"], ranking_right["use_breakout_quality_ranking"]),
            tuple(ranking_right[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS),
        ),
    )

    baseline_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": False},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": False},
    ]
    ranked_rows = [
        {**baseline_rows[0], "use_breakout_quality_ranking": True, "breakout_quality_score": 0.90},
        {**baseline_rows[1], "use_breakout_quality_ranking": True, "breakout_quality_score": 0.40},
    ]
    check(
        "score_ranking_precedes_existing_buy_sort",
        (["B", "A"], ["A", "B"]),
        (
            [row["ticker"] for row in sort_candidate_rows(baseline_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
            [row["ticker"] for row in sort_candidate_rows(ranked_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)],
        ),
    )

    fraction, deployment = calc_projected_capital_metrics(
        proj_cost=150.0, sizing_capital=1000.0, max_position_cap_pct=0.30
    )
    check("capital_deployment_uses_canonical_projected_cost", (0.15, 0.50), (round(fraction, 6), round(deployment, 6)))

    capital_rows = [
        {"ticker": "A", "sort_value": 0.05, "proj_cost": 100.0, "sizing_capital": 1000.0, "max_position_cap_pct": 0.30, "projected_capital_deployment_rate": 1 / 3, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.90, "breakout_quality_rank": {"available": True}},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 200.0, "sizing_capital": 1000.0, "max_position_cap_pct": 0.30, "projected_capital_deployment_rate": 2 / 3, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.78, "breakout_quality_rank": {"available": True}},
        {"ticker": "C", "sort_value": 0.20, "proj_cost": 300.0, "sizing_capital": 1000.0, "max_position_cap_pct": 0.30, "projected_capital_deployment_rate": 1.0, "use_breakout_quality_ranking": True, "breakout_quality_score": 0.68, "breakout_quality_rank": {"available": True}},
    ]
    for policy in (BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED, BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_BUCKET):
        rows = [dict(row, breakout_quality_ranking_policy=policy) for row in capital_rows]
        check(f"{policy}_uses_capital_aware_ordering", ["C", "B", "A"], [row["ticker"] for row in sort_candidate_rows(rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)])
    unavailable_rows = [
        {"ticker": "A", "sort_value": 0.20, "proj_cost": 100.0, "use_breakout_quality_ranking": True, "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED, "breakout_quality_score": None, "breakout_quality_rank": {"available": False}},
        {"ticker": "B", "sort_value": 0.10, "proj_cost": 80.0, "use_breakout_quality_ranking": True, "breakout_quality_ranking_policy": BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED, "breakout_quality_score": None, "breakout_quality_rank": {"available": False}},
    ]
    check("missing_scores_preserve_original_buy_sort_fallback", ["B", "A"], [row["ticker"] for row in sort_candidate_rows(unavailable_rows, method=BUY_LIMIT_OVERAGE_SORT_METHOD)])

    ensemble_rows = []
    for ticker, votes, score in (("A", 2, 0.2), ("B", 1, 0.99), ("C", 2, 0.8)):
        for member_idx in range(votes):
            ensemble_rows.append({
                "ticker": ticker,
                "ensemble_member_key": f"m{member_idx}",
                "params_obj": base,
                "sort_value": 0.0,
                "proj_cost": 100.0,
                "use_breakout_quality_ranking": True,
                "breakout_quality_score": score,
                "breakout_quality_rank": {
                    "score": score,
                    "available": True,
                    "unavailable_reason": "",
                    "score_date": "2025-01-02",
                    "shared_group_score": True,
                    "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
                },
            })
    ensemble_ranked = _aggregate_ensemble_candidate_rows(ensemble_rows, min_agree=1)
    check(
        "ensemble_vote_precedes_score_and_score_breaks_equal_votes",
        [("C", 2), ("A", 2), ("B", 1)],
        [(row["ticker"], row["ensemble_vote_count"]) for row in ensemble_ranked],
    )

    ensemble_payload = build_static_active_param_ensemble_payload(
        members=[
            {"member_index": 1, "seed": 101, "params": params_to_json_dict(base)},
            {"member_index": 2, "seed": 202, "params": params_to_json_dict(replace(base, high_len=205))},
        ],
        random_seed_ensemble={"enabled": True, "seed_count": 2, "min_agree": 2},
        selector="base_finalists_agree",
    )
    ensemble_pair = _build_controlled_param_source_pair(
        {"kind": "static_active_param_ensemble", "payload": ensemble_payload},
        filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
        threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
        fixed_risk=0.01,
    )
    left_members = ensemble_pair[1]["params_ensemble"]
    right_members = ensemble_pair[2]["params_ensemble"]
    check(
        "static_ensemble_preserves_members_and_only_toggles_filter",
        [(201, False, True), (205, False, True)],
        [
            (int(left["params"]["high_len"]), bool(left["params"]["use_breakout_quality_filter"]), bool(right["params"]["use_breakout_quality_filter"]))
            for left, right in zip(left_members, right_members)
        ],
    )

    rolling_payload = {
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
        path = Path(tmp_dir) / "roos_base_best.json"
        path.write_text(json.dumps(rolling_payload), encoding="utf-8")
        rolling_source = _load_param_source(path)
        rolling_pair = _build_controlled_param_source_pair(
            rolling_source,
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            threshold=float(BREAKOUT_QUALITY_DEFAULT_SCORE_THRESHOLD),
            fixed_risk=None,
        )
    check(
        "rolling_schedule_preserves_effective_dates_and_only_toggles_filter",
        (["2021-01-01", "2022-01-01"], False, True, 205),
        (
            sorted(rolling_pair[1]["params_by_effective_date"]),
            rolling_pair[1]["params_by_effective_date"]["2022-01-01"]["use_breakout_quality_filter"],
            rolling_pair[2]["params_by_effective_date"]["2022-01-01"]["use_breakout_quality_filter"],
            rolling_pair[2]["params_by_effective_date"]["2022-01-01"]["high_len"],
        ),
    )

    check(
        "comparison_output_identity_separates_policy_and_ranking_mode",
        True,
        _comparison_output_dir_name(
            COMPARISON_MODE_SCORE_RANKING,
            _comparison_labels(COMPARISON_MODE_SCORE_RANKING),
            param_policy=PARAM_POLICY_BASE_FINALIST_BEST,
            ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_CAPITAL_ADJUSTED,
        )
        != _comparison_output_dir_name(
            COMPARISON_MODE_HARD_FILTER,
            _comparison_labels(COMPARISON_MODE_HARD_FILTER),
            param_policy=PARAM_POLICY_BASE_FINALISTS_AGREE,
        ),
    )

    capacity = _capacity_summary({"portfolio_capacity_rows": [
        {"Orderable_Candidates": 3, "Candidate_Supply_Gap": 7, "End_Position_Gap": 8, "Post_Execution_Positions": 2},
        {"Orderable_Candidates": 0, "Candidate_Supply_Gap": 8, "End_Position_Gap": 8, "Post_Execution_Positions": 2},
    ]})
    check(
        "daily_candidate_and_position_gap_summary",
        (2, 1.5, 2, 16),
        (capacity["sim_day_count"], capacity["avg_orderable_candidates"], capacity["underfilled_end_days"], capacity["end_position_gap_slot_days"]),
    )

    normalized = _normalize_yearly_completeness(pd.DataFrame([
        {"year": 2025, "is_full_year": True, "start_date": "2025-01-02", "end_date": "2025-12-31"},
        {"year": 2026, "is_full_year": True, "start_date": "2026-01-02", "end_date": "2026-03-02"},
    ]))
    check("comparison_partial_final_year_is_not_marked_full", [True, False], list(normalized["is_full_year"]))

    no_filter_history = pd.DataFrame([
        {"Date": "2025-01-03", "Ticker": "A", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-02", "候選日": "2025-01-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-01-10", "Ticker": "A", "Type": "全倉結算(指標)", "成交價": 13.0, "該筆總損益": 3000.0, "R_Multiple": 3.0},
        {"Date": "2025-02-03", "Ticker": "B", "Type": "買進 (新訊號, EV:1.00R)", "買訊日": "2025-01-31", "候選日": "2025-02-03", "進場類型": "normal", "成交價": 10.0},
        {"Date": "2025-02-10", "Ticker": "B", "Type": "全倉結算(停損)", "成交價": 9.0, "該筆總損益": -1000.0, "R_Multiple": -1.0},
    ])
    quality_history = no_filter_history.iloc[:2].copy()
    scores = pd.DataFrame({"ticker": ["A", "B"], "date": ["2025-01-02", "2025-01-31"], SCORE_COLUMN: [0.8, 0.3]}).set_index(["ticker", "date"])[[SCORE_COLUMN]]
    attribution = build_trade_attribution(
        no_filter_trade_history=no_filter_history,
        quality_filter_trade_history=quality_history,
        shared_score_table=scores,
        threshold=0.5,
        no_filter_portfolio_total_r=2.0,
        quality_filter_portfolio_total_r=3.0,
    )
    check_true(
        "trade_attribution_reconciles_portfolio_total_r",
        abs(float(attribution["r_attribution"]["reconciliation_error_r"])) < 1e-12,
    )

    native = _to_json_native({"number": np.float64(1.25), "date": pd.Timestamp("2026-07-26"), "non_finite": np.float64(np.nan)})
    check("comparison_json_payload_is_native_and_strict", {"number": 1.25, "date": "2026-07-26T00:00:00", "non_finite": None}, native)
    check_true("comparison_json_number_is_builtin_float", type(native["number"]) is float)

    # Historical fixed-signal continuation remains a compatibility invariant:
    # re-entry may trigger later, but the original breakout score date/value is inherited.
    reentry_params = replace(
        base,
        use_breakout_reclaim_reentry=True,
        use_breakout_quality_ranking=True,
        breakout_quality_filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    )
    original_rank = {
        "score": 0.731, "available": True, "unavailable_reason": "",
        "score_date": "2021-12-30", "shared_group_score": True,
        "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    }
    position = {
        "ticker": "3706", "entry_type": "normal", "entry_fill_price": 100.0,
        "pure_buy_price": 100.0, "initial_stop": 90.0, "sl": 90.0,
        "qty": 10, "initial_qty": 10,
        "breakout_quality_score": original_rank["score"],
        "breakout_quality_score_date": original_rank["score_date"],
        "breakout_quality_rank": dict(original_rank),
        "use_breakout_quality_ranking": True,
    }
    watch = create_breakout_reentry_watch_state(
        position, exit_date=pd.Timestamp("2022-01-05"), params=reentry_params,
        exit_atr=2.0, exit_qty=10,
    )
    trigger_date = pd.Timestamp("2022-01-17")
    signal = create_breakout_reentry_signal_state(
        watch, close_price=float(watch["confirm_price"]), atr=2.0,
        params=reentry_params, ticker="3706", signal_date=trigger_date,
    )
    inherited = resolve_breakout_quality_rank(signal)
    from filters.breakout_quality.ranking_score_store import SCORE_SOURCE_CANONICAL_RUNTIME
    from filters.breakout_quality.runtime import breakout_quality_ranking_source_context
    with breakout_quality_ranking_source_context(
        score_source=SCORE_SOURCE_CANONICAL_RUNTIME,
        model_architecture=BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
        experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ), patch(
        "core.portfolio_candidates.resolve_breakout_quality_candidate_rank",
        side_effect=AssertionError("fixed-signal re-entry不得重新查trigger-date score"),
    ):
        resolved = _resolve_candidate_quality_ranking(
            params=reentry_params, ticker="3706", signal_date=trigger_date, signal_state=signal
        )
    check(
        "fixed_signal_reentry_inherits_original_breakout_score",
        ("2022-01-17", "2021-12-30", 0.731, "2021-12-30", 0.731),
        (
            pd.Timestamp(signal["signal_date"]).strftime("%Y-%m-%d"),
            inherited["score_date"], inherited["score"], resolved["score_date"], resolved["score"],
        ),
    )

    summary["checks"] = len(results)
    return results, summary


def validate_breakout_quality_binary_dl_param_adaptation_contract_case(_base_params):
    """Current Min-ROOS/Binary-PIT integration contract.

    Historical 4×2 experiment orchestration is recorded in the Experiment Log.
    This validator protects the promoted reusable contracts only: Optimizer owns
    Min training, the compatibility facade has no second implementation, Min
    search fields remain narrow, completed folds are reusable, and PIT runtime
    identity is transported without leakage.
    """
    case_id = "BREAKOUT_QUALITY_BINARY_DL_PARAM_ADAPTATION"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    import filters.breakout_quality.strategy_param_training as compatibility_training
    import services.optimizer.strategy_param_training as canonical_training
    from config.breakout_quality import get_breakout_quality_workflow_settings
    from filters.breakout_quality.binary_pit_score_store import (
        BINARY_PIT_SCORE_SOURCE,
        build_pass_condition_from_binary_point_in_time_scores,
        load_binary_point_in_time_score_table,
    )
    from filters.breakout_quality.runtime import (
        breakout_quality_filter_source_context,
        build_breakout_quality_filter_pass_condition,
    )
    from strategies.breakout.search_space import build_trial_params
    from tools.optimizer.outer_rolling_oos import (
        FOLD_FIXED_STRATEGY_OVERRIDES_KEY,
        _validate_optimizer_runtime_context,
        resolve_optimizer_session_spec_for_fold,
    )

    check_true(
        "min_roos_training_compatibility_facade_delegates_canonical_optimizer_owner",
        compatibility_training.run_param_adaptation_gate is canonical_training.run_param_adaptation_gate
        and compatibility_training.build_min_roos_fold_overrides is canonical_training.build_min_roos_fold_overrides,
    )

    workflow = get_breakout_quality_workflow_settings()
    args = canonical_training._parse_args([])
    base = V16StrategyParams()
    baseline_params = params_to_json_dict(base)
    baseline_params.update({
        "high_len": 220,
        "atr_len": 19,
        "atr_buy_tol": 1.0,
        "atr_times_init": 4.4,
        "atr_times_trail": 3.7,
        "use_history_threshold": False,
        "use_breakout_reclaim_reentry": True,
        "use_kc": True,
    })
    baseline_contract = {
        "meta": {"first_oos_date": "2021-01-01", "last_oos_date": "2021-12-31", "train_window_months": 120, "oos_horizon_months": 12},
        "payload": {"params_ensemble_by_effective_date": {"2021-01-01": [{"member_index": 1, "params": baseline_params}]}},
    }
    p2 = canonical_training.build_min_roos_fold_overrides(
        baseline_contract=baseline_contract, args=args, training_dl_enabled=False
    )
    p3 = canonical_training.build_min_roos_fold_overrides(
        baseline_contract=baseline_contract, args=args, training_dl_enabled=True
    )
    p2_fixed = p2["2021-01-01"]
    p3_fixed = p3["2021-01-01"]
    check(
        "min_roos_fold_contract_is_config_driven_and_dl_state_isolated",
        (tuple(canonical_training.MIN_ROOS_SEARCH_FIELDS), False, True, workflow.experiment_profile),
        (tuple(canonical_training.MIN_ROOS_SEARCH_FIELDS), p2_fixed["use_breakout_quality_filter"], p3_fixed["use_breakout_quality_filter"], args.experiment_profile),
    )
    check_true(
        "min_roos_fixed_overrides_disable_optional_rule_filters",
        all(not p2_fixed[field] for field in OPTIONAL_ENTRY_FILTER_FIELDS)
        and all(p3_fixed.get(k) == v for k, v in canonical_training.ALL_RULE_FILTERS_OFF_OVERRIDES.items()),
    )

    resolved = resolve_optimizer_session_spec_for_fold(
        {FOLD_FIXED_STRATEGY_OVERRIDES_KEY: p3, "fixed_strategy_param_overrides": {"fixed_risk": 0.01}},
        oos_start_date="2021-01-01",
    )
    check_true(
        "fold_runtime_receives_binary_dl_fixed_overrides",
        bool(resolved["fixed_strategy_param_overrides"]["use_breakout_quality_filter"]),
    )

    class _FixedSession:
        optimizer_fixed_tp_percent = 0.0
        def has_fixed_strategy_param(self, name): return name in p2_fixed
        def get_fixed_strategy_param(self, name, default=None): return p2_fixed.get(name, default)
        def resolve_optimizer_tp_percent(self, trial, *, fixed_tp_percent): return fixed_tp_percent

    class _Trial:
        def __init__(self): self.calls = []
        def suggest_int(self, name, low, high, step=1): self.calls.append(name); return int(low)
        def suggest_float(self, name, low, high, step=None): self.calls.append(name); return float(low)
        def suggest_categorical(self, name, choices): self.calls.append(name); return list(choices)[0]

    trial = _Trial()
    trial_params = build_trial_params(_FixedSession(), trial)
    check(
        "min_roos_searches_only_optimizer_owned_trainable_fields",
        set(canonical_training.MIN_ROOS_SEARCH_FIELDS),
        set(trial.calls),
    )
    check_true(
        "fixed_dimensions_do_not_leak_back_into_search",
        not trial_params.use_bb and not trial_params.use_kc and not trial_params.use_breakout_quality_filter,
    )

    with tempfile.TemporaryDirectory() as tmp:
        params_path = Path(tmp) / "roos_base_best.json"
        member = dict(p2_fixed)
        for field in canonical_training.MIN_ROOS_SEARCH_FIELDS:
            member[field] = baseline_params[field]
        params_path.write_text(json.dumps({
            "meta": {
                "first_oos_date": "2021-01-01", "last_oos_date": "2021-12-01",
                "train_window_months": 120, "oos_horizon_months": 12,
                "trials_per_fold": int(args.trials_per_fold),
            },
            "params_ensemble_by_effective_date": {"2021-01-01": [{"member_index": 1, "params": member}]},
        }), encoding="utf-8")
        reused = canonical_training._reuse_existing_min_roos_params_if_compatible(
            prior_preflight={"runtime_identity_sha256": "same"},
            contract={"runtime_identity_sha256": "same"},
            params_path=params_path,
            baseline_contract=baseline_contract,
            fold_overrides=p2,
            args=args,
            training_dl_enabled=False,
            arm_id="P2_HISTORY",
        )
    check_true("completed_min_roos_fold_is_reused_when_runtime_identity_matches", reused is not None)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = root / "manifest.json"
        scores = root / "scores.csv"
        pd.DataFrame({
            "ticker": ["2330", "2330"],
            "date": ["2020-01-02", "2020-01-03"],
            "group_index": [1, 2],
            "dl_quality_score": [0.7, 0.3],
            "fold_id": ["fold_1", "fold_1"],
            "model_information_cutoff": ["2020-01-01", "2020-01-01"],
        }).to_csv(scores, index=False, encoding="utf-8-sig")
        manifest.write_text(json.dumps({
            "schema_type": "binary_point_in_time_scores",
            "score_table": {"row_count": 2},
            "score_period": {"start": "2020-01-02", "end": "2020-01-03"},
        }), encoding="utf-8")
        load_binary_point_in_time_score_table.cache_clear()
        frame = pd.DataFrame({"close": [9.0, 10.0, 11.0]}, index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]))
        candidates = np.array([True, True, True])
        direct = build_pass_condition_from_binary_point_in_time_scores(
            frame, ticker="2330", score_threshold=0.5, candidate_condition=candidates,
            manifest_path=str(manifest), scores_path=str(scores),
        )
        with breakout_quality_filter_source_context(
            score_source=BINARY_PIT_SCORE_SOURCE, manifest_path=str(manifest), scores_path=str(scores)
        ):
            runtime = build_breakout_quality_filter_pass_condition(
                frame, ticker="2330", high_len=220, score_threshold=0.5,
                candidate_condition=candidates, project_root=str(root),
            )

        class _ContextSession:
            def optimizer_runtime_context(self):
                return breakout_quality_filter_source_context(
                    score_source=BINARY_PIT_SCORE_SOURCE, manifest_path=str(manifest), scores_path=str(scores)
                )
        runtime_spec = {"runtime_context_spec": {
            "module": "filters.breakout_quality.runtime",
            "callable": "breakout_quality_filter_source_context",
            "kwargs": {"score_source": BINARY_PIT_SCORE_SOURCE, "manifest_path": str(manifest), "scores_path": str(scores)},
        }}
        _validate_optimizer_runtime_context(_ContextSession(), runtime_spec)
    check("binary_pit_runtime_matches_direct_store_semantics", tuple(direct), tuple(runtime))

    summary["checks"] = len(results)
    return results, summary


def validate_breakout_quality_trade_path_label_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_TRADE_PATH_LABEL"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

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
    check(
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
    check(
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
    check(
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
    check(
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
    check(
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
    check(
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
    check(
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
    check(
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
    check(
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
        check(
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
        check(
            "trade_path_artifact_paths_keep_filter_identity",
            (TRADE_PATH_RESEARCH_FILTER_ID, True),
            (
                            artifact_paths.filter_id,
                            TRADE_PATH_RESEARCH_FILTER_ID in str(artifact_paths.model_dir),
                        ),
        )


    project_root = Path(__file__).resolve().parents[2]
    app_source = (project_root / "services" / "research" / "breakout_quality_application.py").read_text(encoding="utf-8")
    builder_source = (
        project_root / "services" / "breakout_quality" / "trade_path_label_builder.py"
    ).read_text(encoding="utf-8")
    binary_pit_source = (
        project_root / "services" / "breakout_quality" / "binary_point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    check_true(
        "trade_path_menu_completes_model_artifacts_and_strategy_comparison_stays_separate",
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
    check_true(
        "trade_path_builder_reuses_formal_lifecycle_and_never_overwrites_9a",
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
    check_true(
        "trade_path_binary_pit_uses_filter_specific_label_policy",
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
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)
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
    check("retired_research_cli_audit_and_temp_output_are_absent", tuple(True for _ in retired_paths), retired_absent)

    current_paths = (
        "filters/breakout_quality/strategy_comparison.py",
        "filters/breakout_quality/strategy_param_training.py",
        "services/breakout_quality/trade_path_label_builder.py",
        "services/breakout_quality/train.py",
        "services/audit/catalog.py",
        "tools/validate/transient_code_maintenance.py",
    )
    check(
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
    check(
        "required_historical_reconstruction_compatibility_is_preserved",
        tuple(True for _ in compatibility_paths),
        tuple((project_root / relative_path).exists() for relative_path in compatibility_paths),
    )

    application_source = (
        project_root / "services/research/breakout_quality_application.py"
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
    check(
        "retired_research_cli_is_not_routable_or_advertised",
        (False, False),
        (
                    any(token in application_source for token in retired_command_tokens),
                    any(token in current_docs for token in retired_command_tokens),
                ),
    )

    from tools.validate.transient_code_maintenance import summarize_transient_code_maintenance

    # The full repository maintenance scan is already executed once by meta quality.
    # Here we only exercise its output contract on a tiny isolated tree so the
    # consistency step does not duplicate the whole-project AST/source scan.
    with tempfile.TemporaryDirectory(prefix="maintenance_contract_fixture_") as temp_dir:
        fixture_root = Path(temp_dir)
        (fixture_root / "doc").mkdir(parents=True, exist_ok=True)
        (fixture_root / "tools" / "validate").mkdir(parents=True, exist_ok=True)
        (fixture_root / "doc" / "TEST_SUITE_CHECKLIST.md").write_text(
            "# fixture\n",
            encoding="utf-8",
        )
        maintenance = summarize_transient_code_maintenance(fixture_root)
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
    retired_dedicated_tests = maintenance.get("retired_dedicated_tests")
    private_zero_callers = maintenance.get("private_zero_callers")
    experiment_execution_branches = maintenance.get("experiment_execution_branches")
    growth_reviews = maintenance.get("growth_reviews")
    source_shape_inventory = maintenance.get("source_shape_inventory")
    growth = maintenance.get("growth")
    maintenance_lists = (
        maintenance.get("stale_modules"),
        maintenance.get("disabled_audits"),
        maintenance.get("oversized_transient_tests"),
        retired_dedicated_tests,
        private_zero_callers,
        experiment_execution_branches,
        growth_reviews,
    )
    maintenance_category_count = (
        sum(len(items) for items in maintenance_lists)
        if all(isinstance(items, list) for items in maintenance_lists)
        else None
    )
    check(
        "transient_code_maintenance_scan_is_advisory_and_internally_consistent",
        (True, True, True, True, True, True, True, True),
        (
                    maintenance_count_valid,
                    maintenance.get("status") == maintenance_expected_status,
                    maintenance.get("needs_slimming") == bool(maintenance_candidate_count)
                    if maintenance_count_valid
                    else False,
                    maintenance.get("advisory_only") is True,
                    isinstance(retired_dedicated_tests, list),
                    maintenance_category_count == maintenance_candidate_count
                    if maintenance_count_valid and maintenance_category_count is not None
                    else False,
                    isinstance(source_shape_inventory, dict)
                    and type(source_shape_inventory.get("line_count")) is int
                    and type(source_shape_inventory.get("validator_count")) is int,
                    isinstance(growth, dict)
                    and isinstance(growth.get("review_reasons"), list),
                ),
    )

    summary["retired_paths"] = list(retired_paths)
    summary["current_replacements"] = list(current_paths)
    summary["compatibility_preserved"] = list(compatibility_paths)
    summary["maintenance_contract_fixture"] = maintenance
    return results, summary

def validate_breakout_quality_strategy_readable_report_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_STRATEGY_READABLE_REPORT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)
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
    check_true(
        "all_formal_strategy_results_have_persistent_markdown_simple_report",
        all(row[1] for row in contract_rows),
    )
    check_true(
        "all_formal_strategy_results_have_console_readable_renderer",
        all(row[2] and row[3] for row in contract_rows),
    )

    comparison_source = (
        project_root / "filters/breakout_quality/strategy_comparison.py"
    ).read_text(encoding="utf-8")
    reporting_source = (
        project_root / "filters/breakout_quality/strategy_compare_reporting.py"
    ).read_text(encoding="utf-8")
    check_true(
        "strategy_compare_run_and_reuse_share_canonical_pair_readable_report_renderer",
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
    continuous_ranker_source = (
        project_root / "services/breakout_quality/train_continuous_ranker.py"
    ).read_text(encoding="utf-8")
    training_progress_source = (project_root / "core/training_progress.py").read_text(
        encoding="utf-8"
    )
    strategy_training_source = (
        project_root / "services/research/strategy_compare_training.py"
    ).read_text(encoding="utf-8")
    strategy_app_service_source = (
        project_root / "services/research/breakout_quality_application.py"
    ).read_text(encoding="utf-8")
    strategy_reuse_source = (
        project_root / "filters/breakout_quality/strategy_compare_reuse.py"
    ).read_text(encoding="utf-8")
    strategy_replay_source = (
        project_root / "filters/breakout_quality/strategy_compare_replay.py"
    ).read_text(encoding="utf-8")
    strategy_engine_source = (
        project_root / "filters/breakout_quality/strategy_compare_engine.py"
    ).read_text(encoding="utf-8")
    model_report_source = (
        project_root / "services/breakout_quality/report.py"
    ).read_text(encoding="utf-8")
    strategy_dashboard_source = (project_root / "core/strategy_dashboard.py").read_text(encoding="utf-8")
    optimizer_callbacks_source = (project_root / "services/optimizer/callbacks.py").read_text(encoding="utf-8")
    outer_roos_source = (project_root / "services/optimizer/outer_rolling_oos.py").read_text(encoding="utf-8")
    render_report_source = comparison_source[
        comparison_source.index("def render_strategy_aggregate_report("):
        comparison_source.index("def _run_directory(")
    ]
    research_report_contract_source = (project_root / "core/research_report_contract.py").read_text(encoding="utf-8")
    check_true(
        "strategy_compare_main_report_is_fixed_sop_without_hypothesis_specific_safety_curve",
        '"strategy_diagnostics.md"' in comparison_source
                and "Core Performance" in research_report_contract_source
                and "Trade Quality / MFE × Safety" in research_report_contract_source
                and "Selection Quality" in research_report_contract_source
                and "Capital / Execution" in research_report_contract_source
                and "Execution Summary" in research_report_contract_source
                and "Raw Safety Gate Sensitivity" not in render_report_source
                and "render_safety_gate_sensitivity_table" not in render_report_source
                and "render_strategy_run_execution_table" in render_report_source
                and "報表分工" not in render_report_source
                and "render_strategy_selection_quality_table" in render_report_source
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
                and '"top_target_r": finite_number(metrics.get("top_decile_target_mean"))' in diagnostics_source
                and '"bottom_target_r": finite_number(metrics.get("bottom_decile_target_mean"))' in diagnostics_source
                and "from core.report_style import best_worst_signals, finite_number, styled_signal" in diagnostics_source
                and "def _finite(" not in diagnostics_source
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
    check_true(
        "strategy_compare_progress_and_final_report_share_high_level_wall_time_summary",
        "arm_execution_timing" in comparison_source
                and '"execution_summary": execution_summary' in comparison_source
                and 'f"[RUN] {on_arm.arm_id} {on_arm.name} "' in comparison_source
                and 'f"[DONE] {on_arm.arm_id} {on_arm.name} "' in comparison_source
                and 'elapsed={format_elapsed(' in comparison_source
                and 'total={format_elapsed(' in comparison_source
                and 'section_contract("strategy.standard_sop", "execution_summary").display_title' in comparison_source
                and "render_strategy_run_execution_table" in comparison_source
                and "_selector_timing_table(" not in render_report_source,
    )

    strategy_compare_config_source = (project_root / "config/strategy_compare.py").read_text(encoding="utf-8")
    point_in_time_source = (
        project_root / "services/breakout_quality/point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    from config.strategy_compare import get_strategy_multi_seed_robustness_settings
    from filters.breakout_quality.strategy_multi_seed_robustness import (
        _benchmark_identity_payload,
        _scientific_benchmark_parameter_identities,
        _build_durable_result_artifacts,
        _model_artifact_identity_payload,
        _validate_scientific_observation_manifest,
        _strategy_only_baseline_action,
        SCIENTIFIC_DURABLE_RESULT_KEYS,
        SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME,
        MEAN_METRICS,
        _strategy_only_baseline_context_available,
        _validate_durable_result_artifacts,
        _validate_seed_result_scientific_identities,
        _validate_seed_yearly_results_frame,
        _write_scientific_observation_manifest,
        _write_seed_results,
        _write_seed_yearly_results,
    )
    from config.training_policy import (
        get_robustness_benchmark_policy_snapshot,
        get_strategy_parameter_training_policy_snapshot,
    )
    from core.strategy_param_artifacts import (
        STRATEGY_PARAM_WORK_SCOPE_BENCHMARK,
        STRATEGY_PARAM_SCIENTIFIC_IDENTITY_SCHEMA,
        compute_strategy_param_file_sha256,
        compute_strategy_param_scientific_sha256,
        resolve_strategy_param_optimizer_work_dir,
    )
    from services.optimizer.strategy_param_service import (
        _benchmark_build_contract_scientific_payload,
        _benchmark_fast_republish_source,
        _benchmark_manifest_matches_current_policy,
        _benchmark_schedule_build_contract,
        ensure_robustness_benchmark_strategy_parameter_artifact,
    )
    from filters.breakout_quality.strategy_compare_sources import (
        strategy_param_source_identity_sha256,
    )

    check(
        "robustness_strategy_only_completed_result_without_pending_model_context_is_true_reuse",
        "REUSE",
        _strategy_only_baseline_action(
            scientific_result_exists=True,
            baseline_context_required=False,
            baseline_context_available=False,
        ),
    )
    check(
        "robustness_strategy_only_missing_transient_baseline_context_rebuilds_context_only",
        "REBUILD_CONTEXT",
        _strategy_only_baseline_action(
            scientific_result_exists=True,
            baseline_context_required=True,
            baseline_context_available=False,
        ),
    )
    check(
        "robustness_strategy_only_missing_scientific_result_runs_scientific_replay",
        "RUN_SCIENTIFIC",
        _strategy_only_baseline_action(
            scientific_result_exists=False,
            baseline_context_required=True,
            baseline_context_available=False,
        ),
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        missing_context_available = _strategy_only_baseline_context_available(
            output_dir=Path(temp_dir) / "missing_baseline",
            settings=SimpleNamespace(
                dataset="full", param_policy="base-finalist-best",
                max_positions=10, rotation="off",
            ),
            arm=SimpleNamespace(
                arm_id="C61", param_source="full_rolling",
                param_policy="base-finalist-best", rule_policy="all_off",
            ),
            binding={
                "sha256": "0" * 64,
                "evaluation_mode": "rolling",
            },
            comparison_start="2021-01-01",
            comparison_end="2026-03-02",
        )
    check_true(
        "robustness_missing_transient_baseline_json_is_cache_miss_not_runtime_failure",
        missing_context_available is False,
    )
    model_identity_scientific = {
        "robustness_id": "extending_window_rolling",
        "profile_id": "strategy_compare",
        "dataset": "full",
        "dataset_identity": {"sha256": "dataset-a"},
        "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
        "benchmark_id": "end_to_end_v1",
        "resolved_seeds": [11, 22],
        "training_defaults": {"epochs": 200},
        "benchmark_parameter_artifact_identities": {
            "C59:seed=11": {"sha256": "param-old"}
        },
        "stochastic_arms": [
            {
                "arm_id": "C59",
                "model_seed_sensitive": True,
                "runtime_dl_sources": [
                    {
                        "dl_id": "CONT13E_ROLL",
                        "filter_id": "breakout_quality_v1",
                        "model_architecture": "inception_time_v1",
                        "experiment_profile": "profile_e",
                    }
                ],
            }
        ],
    }
    model_identity_before = _model_artifact_identity_payload(model_identity_scientific)
    model_identity_scientific["benchmark_parameter_artifact_identities"] = {
        "C59:seed=11": {"sha256": "param-new"}
    }
    model_identity_after = _model_artifact_identity_payload(model_identity_scientific)
    check_true(
        "robustness_model_artifact_identity_is_independent_of_strategy_param_retraining",
        model_identity_before == model_identity_after
        and 'contract.get("model_artifact_fingerprint") or contract["fingerprint"]' in multi_seed_source
        and "reusable_artifact_roots_exist = model_dir.is_dir()" in multi_seed_source,
    )
    seed_expansion_before = dict(model_identity_scientific)
    seed_expansion_before["resolved_seeds"] = [11, 22]
    seed_expansion_after = dict(seed_expansion_before)
    seed_expansion_after["resolved_seeds"] = [11, 22, 33, 44]
    check_true(
        "robustness_model_artifact_identity_is_independent_of_seed_count_expansion",
        _model_artifact_identity_payload(seed_expansion_before)
        == _model_artifact_identity_payload(seed_expansion_after)
        and '"resolved_seeds": list(scientific.get("resolved_seeds") or ())' not in multi_seed_source,
    )

    benchmark_binding_a = {
        ("C61", 11): {
            "family": "full", "evaluation_mode": "rolling",
            "param_policy": "base-finalist-best", "sha256": "same-param",
            "manifest_sha256": "manifest-a",
        }
    }
    benchmark_binding_b = {
        ("C61", 11): {
            **benchmark_binding_a[("C61", 11)],
            "manifest_sha256": "manifest-b",
        }
    }
    check_true(
        "robustness_benchmark_manifest_publication_sha_does_not_change_scientific_param_identity",
        _scientific_benchmark_parameter_identities(_benchmark_identity_payload(benchmark_binding_a))
        == _scientific_benchmark_parameter_identities(_benchmark_identity_payload(benchmark_binding_b)),
    )
    check_true(
        "robustness_cross_fingerprint_completed_result_migration_is_intentionally_unsupported",
        "_import_seed_expansion_results(" not in multi_seed_source
        and "_seed_expansion_source_run(" not in multi_seed_source
        and "[COMPATIBLE RESULT REUSE]" not in multi_seed_source
        and "[SEED EXPANSION REUSE]" not in multi_seed_source,
    )

    benchmark = get_robustness_benchmark_policy_snapshot()
    benchmark_seed = int(benchmark["resolved_seeds"][0])
    benchmark_family = "min"
    benchmark_policy = "base_finalist_best"
    benchmark_training = get_strategy_parameter_training_policy_snapshot(
        evaluation_mode="rolling"
    )
    benchmark_build_contract = _benchmark_schedule_build_contract(
        benchmark_id=str(benchmark["benchmark_id"]),
        seed=benchmark_seed,
        family=benchmark_family,
        policy=benchmark_policy,
        dataset="full",
        max_positions=10,
        rotation="off",
        fixed_risk=0.01,
        max_position_cap_pct=20.0,
    )
    benchmark_agree_contract = _benchmark_schedule_build_contract(
        benchmark_id=str(benchmark["benchmark_id"]),
        seed=benchmark_seed,
        family=benchmark_family,
        policy="base_finalists_agree",
        dataset="full",
        max_positions=10,
        rotation="off",
        fixed_risk=0.01,
        max_position_cap_pct=20.0,
    )
    check_true(
        "robustness_best_and_finalists_agree_share_one_optimizer_search_contract",
        _benchmark_build_contract_scientific_payload(benchmark_build_contract)
        == _benchmark_build_contract_scientific_payload(benchmark_agree_contract)
        and benchmark_build_contract["policy"] != benchmark_agree_contract["policy"],
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        optimizer_calls = {"count": 0}

        def fake_full_benchmark_optimizer(**kwargs):
            optimizer_calls["count"] += 1
            active_dir = Path(kwargs["output_relative_dir"]) / "active_params"
            active_dir.mkdir(parents=True, exist_ok=True)
            for selector, filename in (
                ("base_finalist_best", "roos_base_best.json"),
                ("base_finalists_agree", "roos_base_finalists_agree.json"),
            ):
                payload = {
                    "schema_type": "active_param_ensemble",
                    "schema_version": 1,
                    "mode": "rolling",
                    "selector": selector,
                    "params_ensemble_by_effective_date": {
                        "2021-01-01": [
                            {
                                "member_index": 1,
                                "seed": benchmark_seed,
                                "params": {"high_len": 201, "atr_len": 14},
                            }
                        ]
                    },
                    "meta": {"last_oos_date": "2026-03-02"},
                    "summary": {"oos_end_date": "2026-03-02"},
                }
                (active_dir / filename).write_text(json.dumps(payload), encoding="utf-8")
            return {"params_path": active_dir / "roos_base_best.json"}

        ensure_kwargs = {
            "benchmark_id": str(benchmark["benchmark_id"]),
            "seed": benchmark_seed,
            "family": "full",
            "evaluation_mode": "rolling",
            "comparison_end_date": "2026-03-02",
            "dataset": "full",
            "filter_id": BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            "model_architecture": BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
            "experiment_profile": UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
            "max_positions": 10,
            "rotation": "off",
            "fixed_risk": 0.01,
            "max_position_cap_pct": 20.0,
            "quiet": True,
        }
        with patch(
            "services.optimizer.strategy_param_training.prepare_selection_historical_full_roos_params",
            side_effect=fake_full_benchmark_optimizer,
        ):
            best_result = ensure_robustness_benchmark_strategy_parameter_artifact(
                temp_root, policy="base-finalist-best", **ensure_kwargs
            )
            agree_result = ensure_robustness_benchmark_strategy_parameter_artifact(
                temp_root, policy="base-finalists-agree", **ensure_kwargs
            )
        benchmark_manifest = json.loads(
            Path(agree_result["manifest_path"]).read_text(encoding="utf-8")
        )
        check_true(
            "robustness_same_seed_family_search_publishes_best_and_agree_once",
            optimizer_calls["count"] == 1
            and str(best_result["action"]) in {"BUILD", "REBUILD"}
            and str(agree_result["action"]) == "REUSE"
            and {"base_finalist_best", "base_finalists_agree"}.issubset(
                set(benchmark_manifest.get("artifacts") or {})
            ),
        )
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        target = temp_root / "min_base_best.json"
        target_payload = {
            "schema_type": "rolling_oos_param_set",
            "schema_version": 1,
            "selector": benchmark_policy,
            "created_at": "2026-08-22T12:00:00+08:00",
            "meta": {"last_oos_date": "2026-03-02", "diagnostic": "a"},
            "params_by_effective_date": {
                "2021-01-01": {"high_len": 201, "atr_len": 14},
                "2022-01-01": {"high_len": 205, "atr_len": 14},
            },
            "folds": [
                {"effective_start": "2021-01-01", "effective_end": "2021-12-31"},
                {"effective_start": "2022-01-01", "effective_end": "2026-03-02"},
            ],
        }
        target.write_text(json.dumps(target_payload), encoding="utf-8")
        target_sha = compute_strategy_param_file_sha256(target)
        target_scientific_sha = compute_strategy_param_scientific_sha256(target)
        manifest_path = temp_root / "min_manifest.json"
        manifest = {
            "benchmark_id": str(benchmark["benchmark_id"]),
            "benchmark_seed": benchmark_seed,
            "family": benchmark_family,
            "evaluation_mode": "schedule",
            # Deliberately pin an old N=1 membership snapshot. Membership expansion
            # must not invalidate this concrete seed when fitting inputs are unchanged.
            "benchmark": {
                "benchmark_id": str(benchmark["benchmark_id"]),
                "seed_count": 1,
                "resolved_seeds": [benchmark_seed],
            },
            "training_policy": {
                "optimizer_seed": benchmark_seed,
                "trials_per_fold": int(benchmark_training["trials_per_fold"]),
                "benchmark_seed_override": True,
                "evaluation_mode": "schedule",
            },
            "artifacts": {
                benchmark_policy: {
                    "sha256": target_sha,
                    "scientific_sha256": target_scientific_sha,
                    "source": {
                        "build_contract": benchmark_build_contract,
                        "schedule_kind": "annual_refit",
                    },
                }
            },
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        membership_only_reuse = _benchmark_manifest_matches_current_policy(
            manifest_path,
            benchmark=benchmark,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
            evaluation_mode="rolling",
            policy=benchmark_policy,
            target_path=target,
            comparison_end_date="2026-03-02",
            build_contract=benchmark_build_contract,
        )
        stale_contract = dict(benchmark_build_contract)
        stale_contract["trials_per_fold"] = int(stale_contract["trials_per_fold"]) + 1
        stale_contract_rejected = not _benchmark_manifest_matches_current_policy(
            manifest_path,
            benchmark=benchmark,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
            evaluation_mode="rolling",
            policy=benchmark_policy,
            target_path=target,
            comparison_end_date="2026-03-02",
            build_contract=stale_contract,
        )
        tampered_payload = json.loads(json.dumps(target_payload))
        tampered_payload["params_by_effective_date"]["2022-01-01"]["high_len"] = 210
        target.write_text(json.dumps(tampered_payload), encoding="utf-8")
        stale_sha_rejected = not _benchmark_manifest_matches_current_policy(
            manifest_path,
            benchmark=benchmark,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
            evaluation_mode="rolling",
            policy=benchmark_policy,
            target_path=target,
            comparison_end_date="2026-03-02",
            build_contract=benchmark_build_contract,
        )
    check_true(
        "robustness_per_seed_param_reuse_ignores_membership_but_rejects_stale_contract_or_sha",
        membership_only_reuse and stale_contract_rejected and stale_sha_rejected,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        fast_target = temp_root / "models" / "strategy_params" / "benchmark" / "x" / "min_base_best_seed.json"
        fast_target.parent.mkdir(parents=True, exist_ok=True)
        fast_payload = {
            "schema_type": "rolling_oos_param_set",
            "schema_version": 1,
            "selector": benchmark_policy,
            "created_at": "2026-08-22T12:00:00+08:00",
            "meta": {"last_oos_date": "2026-03-02", "diagnostic": "target"},
            "params_by_effective_date": {
                "2021-01-01": {"high_len": 201, "atr_len": 14}
            },
            "folds": [
                {"effective_start": "2021-01-01", "effective_end": "2026-03-02"}
            ],
        }
        fast_target.write_text(json.dumps(fast_payload), encoding="utf-8")
        fast_work_root = resolve_strategy_param_optimizer_work_dir(
            temp_root,
            scope=STRATEGY_PARAM_WORK_SCOPE_BENCHMARK,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
        )
        fast_source = (
            fast_work_root
            / "schedule_2021_forward"
            / "p2_dl_off_trained"
            / "active_params"
            / "roos_base_best.json"
        )
        fast_source.parent.mkdir(parents=True, exist_ok=True)
        fast_source_payload = json.loads(json.dumps(fast_payload))
        fast_source_payload["created_at"] = "2026-08-22T13:00:00+08:00"
        fast_source_payload["meta"]["diagnostic"] = "workspace"
        fast_source.write_text(json.dumps(fast_source_payload), encoding="utf-8")
        retained_contract = dict(benchmark_build_contract)
        retained_contract["schema"] = "robustness_strategy_schedule_work_v2"
        (fast_work_root / "work_contract.json").write_text(
            json.dumps(retained_contract), encoding="utf-8"
        )
        publication_only_repair = _benchmark_fast_republish_source(
            root=temp_root,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
            policy=benchmark_policy,
            target_path=fast_target,
            build_contract=benchmark_build_contract,
            comparison_end_date="2026-03-02",
        )
        changed_fit_contract = dict(benchmark_build_contract)
        changed_fit_contract["trials_per_fold"] = int(changed_fit_contract["trials_per_fold"]) + 1
        scientific_change_not_repairable = _benchmark_fast_republish_source(
            root=temp_root,
            benchmark_id=str(benchmark["benchmark_id"]),
            seed=benchmark_seed,
            family=benchmark_family,
            policy=benchmark_policy,
            target_path=fast_target,
            build_contract=changed_fit_contract,
            comparison_end_date="2026-03-02",
        )
        publication_raw_differs_but_scientific_matches = (
            compute_strategy_param_file_sha256(fast_target)
            != compute_strategy_param_file_sha256(fast_source)
            and compute_strategy_param_scientific_sha256(fast_target)
            == compute_strategy_param_scientific_sha256(fast_source)
        )
    check_true(
        "robustness_benchmark_publication_only_stale_uses_fast_repair_not_optimizer_resume",
        publication_only_repair == fast_source.resolve()
        and scientific_change_not_repairable is None,
    )

    check_true(
        "strategy_param_runtime_identity_ignores_created_at_and_diagnostic_metadata",
        publication_raw_differs_but_scientific_matches,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "runtime_source.json"
        runtime_payload = json.loads(json.dumps(target_payload))
        temp_path.write_text(json.dumps(runtime_payload), encoding="utf-8")
        compare_sha_a = strategy_param_source_identity_sha256(
            {"kind": "rolling_oos_param_schedule", "payload": runtime_payload},
            source_path=temp_path,
        )
        runtime_payload["created_at"] = "2026-08-22T14:00:00+08:00"
        runtime_payload["meta"]["diagnostic"] = "compare-refresh"
        temp_path.write_text(json.dumps(runtime_payload), encoding="utf-8")
        compare_sha_b = strategy_param_source_identity_sha256(
            {"kind": "rolling_oos_param_schedule", "payload": runtime_payload},
            source_path=temp_path,
        )
        runtime_payload["params_by_effective_date"]["2022-01-01"]["high_len"] = 211
        compare_sha_c = strategy_param_source_identity_sha256(
            {"kind": "rolling_oos_param_schedule", "payload": runtime_payload},
            source_path=temp_path,
        )
    check_true(
        "strategy_compare_param_identity_reuses_runtime_scientific_hash_not_publication_bytes",
        compare_sha_a == compare_sha_b and compare_sha_b != compare_sha_c,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        left = temp_root / "ensemble_left.json"
        right = temp_root / "ensemble_right.json"
        ensemble_payload = {
            "schema_type": "active_param_ensemble",
            "schema_version": 1,
            "mode": "rolling",
            "selector": benchmark_policy,
            "created_at": "2026-08-22T12:00:00+08:00",
            "random_seed_ensemble": {"seed_count": 2, "min_agree": 2},
            "meta": {"last_oos_date": "2026-03-02", "diagnostic": "left"},
            "params_ensemble_by_effective_date": {
                "2021-01-01": [
                    {"member_index": 1, "seed": 101, "score": 1.1, "params": {"high_len": 201}},
                    {"member_index": 2, "seed": 202, "score": 2.2, "params": {"high_len": 205}},
                ]
            },
            "folds": [
                {"effective_start": "2021-01-01", "effective_end": "2026-03-02"}
            ],
        }
        left.write_text(json.dumps(ensemble_payload), encoding="utf-8")
        right_payload = json.loads(json.dumps(ensemble_payload))
        right_payload["created_at"] = "2026-08-22T13:00:00+08:00"
        right_payload["meta"]["diagnostic"] = "right"
        right_payload["params_ensemble_by_effective_date"]["2021-01-01"][0]["score"] = 99.0
        right.write_text(json.dumps(right_payload), encoding="utf-8")
        stable_ensemble_identity = (
            compute_strategy_param_scientific_sha256(left)
            == compute_strategy_param_scientific_sha256(right)
        )
        right_payload["params_ensemble_by_effective_date"]["2021-01-01"][0]["params"]["high_len"] = 202
        right.write_text(json.dumps(right_payload), encoding="utf-8")
        runtime_ensemble_change_detected = (
            compute_strategy_param_scientific_sha256(left)
            != compute_strategy_param_scientific_sha256(right)
        )
    check_true(
        "strategy_param_runtime_identity_tracks_ensemble_params_not_optimizer_member_provenance",
        stable_ensemble_identity and runtime_ensemble_change_detected,
    )

    identity_frame = pd.DataFrame([{
        "arm_id": "C59",
        "seed": 11,
        "strategy_param_sha256": "param-sha",
        "strategy_param_manifest_sha256": "manifest-sha",
        "model_sha256": "model-sha",
        "score_sha256": "score-sha",
        "runtime_source_identity_sha256": "runtime-sha",
    }])
    identity_contract = {
        "benchmark_id": "end_to_end_v1",
        "benchmark_parameter_artifact_identities": {
            "C59:seed=11": {
                "sha256": "param-sha",
            }
        },
        "model_seed_sensitive_arm_ids": ["C59"],
    }
    _validate_seed_result_scientific_identities(identity_frame, contract=identity_contract)

    pit_identity_frame = identity_frame.copy()
    pit_identity_frame.loc[0, "model_sha256"] = ""
    pit_identity_frame.loc[0, "fold_count"] = 6
    pit_identity_contract = {
        **identity_contract,
        "stochastic_arms": [{
            "arm_id": "C59",
            "runtime_dl_sources": [{"score_source": "selection_point_in_time"}],
        }],
    }
    pit_retained_observation_accepted = True
    try:
        _validate_seed_result_scientific_identities(
            pit_identity_frame, contract=pit_identity_contract
        )
    except ValueError:
        pit_retained_observation_accepted = False
    pit_missing_score_rejected = False
    try:
        invalid_pit_frame = pit_identity_frame.copy()
        invalid_pit_frame.loc[0, "score_sha256"] = ""
        _validate_seed_result_scientific_identities(
            invalid_pit_frame, contract=pit_identity_contract
        )
    except ValueError:
        pit_missing_score_rejected = True
    pit_missing_fold_count_rejected = False
    try:
        invalid_pit_frame = pit_identity_frame.copy()
        invalid_pit_frame.loc[0, "fold_count"] = None
        _validate_seed_result_scientific_identities(
            invalid_pit_frame, contract=pit_identity_contract
        )
    except ValueError:
        pit_missing_fold_count_rejected = True
    check_true(
        "robustness_pit_completed_observation_survives_checkpoint_retention_cleanup",
        pit_retained_observation_accepted
        and pit_missing_score_rejected
        and pit_missing_fold_count_rejected,
    )

    stale_identity_rejected = False
    try:
        stale_frame = identity_frame.copy()
        stale_frame.loc[0, "strategy_param_sha256"] = "different"
        _validate_seed_result_scientific_identities(stale_frame, contract=identity_contract)
    except ValueError:
        stale_identity_rejected = True
    check_true(
        "robustness_completed_observation_reuse_rejects_stale_strategy_param_identity",
        stale_identity_rejected,
    )
    refreshed_manifest_frame = identity_frame.copy()
    refreshed_manifest_frame.loc[0, "strategy_param_manifest_sha256"] = "new-publication-manifest-sha"
    manifest_refresh_accepted = True
    try:
        _validate_seed_result_scientific_identities(
            refreshed_manifest_frame, contract=identity_contract
        )
    except ValueError:
        manifest_refresh_accepted = False
    missing_manifest_rejected = False
    try:
        missing_manifest_frame = identity_frame.copy()
        missing_manifest_frame.loc[0, "strategy_param_manifest_sha256"] = ""
        _validate_seed_result_scientific_identities(
            missing_manifest_frame, contract=identity_contract
        )
    except ValueError:
        missing_manifest_rejected = True
    check_true(
        "robustness_manifest_sha_is_provenance_not_replay_scientific_identity",
        manifest_refresh_accepted and missing_manifest_rejected,
    )

    yearly_arm = SimpleNamespace(arm_id="C59")
    partial_yearly = pd.DataFrame([{
        "arm_id": "C59", "seed": 11, "arm_order": 1, "seed_order": 1,
        "year": 2021, "return_pct": 1.0, "is_complete_year": True,
    }])
    partial_yearly_rejected = False
    try:
        _validate_seed_yearly_results_frame(
            partial_yearly, stochastic_arms=(yearly_arm,), seeds=(11,),
            expected_years=(2021, 2022), require_complete_units=True,
        )
    except ValueError:
        partial_yearly_rejected = True
    complete_yearly = pd.concat([
        partial_yearly,
        pd.DataFrame([{
            "arm_id": "C59", "seed": 11, "arm_order": 1, "seed_order": 1,
            "year": 2022, "return_pct": 2.0, "is_complete_year": True,
        }]),
    ], ignore_index=True)
    complete_yearly_accepted = True
    try:
        _validate_seed_yearly_results_frame(
            complete_yearly, stochastic_arms=(yearly_arm,), seeds=(11,),
            expected_years=(2021, 2022), require_complete_units=True,
        )
    except ValueError:
        complete_yearly_accepted = False
    check_true(
        "robustness_resume_requires_complete_year_set_before_unit_reuse",
        partial_yearly_rejected and complete_yearly_accepted,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        durable_root = Path(temp_dir)
        for filename in (
            "seed_results.csv", "seed_yearly_returns.csv",
            "robustness_summary.json", "robustness_report.md",
        ):
            (durable_root / filename).write_text(f"stable:{filename}\n", encoding="utf-8")
        durable_manifest = _build_durable_result_artifacts(durable_root)
        durable_before = _validate_durable_result_artifacts(durable_root, durable_manifest)
        (durable_root / "robustness_report.md").write_text("tampered\n", encoding="utf-8")
        durable_after = _validate_durable_result_artifacts(durable_root, durable_manifest)
        scientific_after_report_refresh = _validate_durable_result_artifacts(
            durable_root,
            durable_manifest,
            keys=SCIENTIFIC_DURABLE_RESULT_KEYS,
        )
    check_true(
        "robustness_completed_run_reuse_requires_durable_result_sha_integrity",
        durable_before and not durable_after,
    )
    check_true(
        "robustness_scientific_observation_integrity_is_independent_of_derived_report_refresh",
        scientific_after_report_refresh
        and "SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME" in multi_seed_source
        and "_validate_scientific_observation_manifest(" in multi_seed_source,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        evidence_root = Path(temp_dir) / "scientific-evidence-test"
        evidence_root.mkdir(parents=True, exist_ok=True)
        evidence_contract = {
            "fingerprint": evidence_root.name,
            "profile_id": "extending_window_rolling",
            "benchmark_id": "end_to_end_v1",
            "comparison_period": {"start": "2021-01-01", "end": "2021-12-31"},
            "resolved_seeds": [11],
            "stochastic_arms": [{"arm_id": "C59"}],
            "model_seed_sensitive_arm_ids": [],
            "benchmark_parameter_artifact_identities": {
                "C59:seed=11": {"sha256": "param-sha"},
            },
            "retention": {"keep_attribution_source": False},
        }
        seed_row = {
            "arm_id": "C59",
            "seed": 11,
            "arm_order": 1,
            "seed_order": 1,
            "strategy_param_sha256": "param-sha",
            "strategy_param_manifest_sha256": "manifest-provenance",
        }
        seed_row.update({key: 1.0 for _label, key, _unit in MEAN_METRICS})
        evidence_seed_frame = pd.DataFrame([seed_row])
        evidence_yearly_frame = pd.DataFrame([{
            "arm_id": "C59", "seed": 11, "arm_order": 1, "seed_order": 1,
            "year": 2021, "return_pct": 1.0, "is_complete_year": True,
        }])
        _write_seed_results(evidence_root / "seed_results.csv", evidence_seed_frame)
        _write_seed_yearly_results(
            evidence_root / "seed_yearly_returns.csv", evidence_yearly_frame
        )
        _write_scientific_observation_manifest(
            evidence_root,
            contract=evidence_contract,
            seed_frame=evidence_seed_frame,
            seed_yearly_frame=evidence_yearly_frame,
            attribution_index_path=None,
        )
        # A later failed presentation/resume attempt may rewrite lifecycle
        # manifest.json, but must not revoke already committed observations.
        (evidence_root / "manifest.json").write_text(
            json.dumps({"status": "FAILED", "contract": evidence_contract}),
            encoding="utf-8",
        )
        lifecycle_independent_evidence = _validate_scientific_observation_manifest(
            evidence_root
        )
        # Without the exact-fingerprint READY marker, no legacy/cross-fingerprint
        # inference is attempted; the current scientific identity must rebuild.
        (evidence_root / SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME).unlink()
        missing_ready_marker = _validate_scientific_observation_manifest(evidence_root)
    check_true(
        "robustness_scientific_observation_commit_is_exact_fingerprint_only",
        lifecycle_independent_evidence is not None
        and missing_ready_marker is None
        and SCIENTIFIC_OBSERVATIONS_MANIFEST_FILENAME in multi_seed_source
        and "_load_legacy_scientific_observations_for_backfill(" not in multi_seed_source
        and "_write_scientific_observation_manifest(" in multi_seed_source
        and "_validate_scientific_observation_manifest(" in multi_seed_source,
    )

    check_true(
        "robustness_completed_rerun_is_true_noop_and_interrupted_fixed_baseline_can_resume",
        "completed_run = _validate_completed_run(" in multi_seed_source
        and "[ROBUSTNESS REUSE]" in multi_seed_source
        and "_load_local_fixed_baseline_if_compatible(" in multi_seed_source
        and "inspect_robustness_benchmark_strategy_parameter_artifact(" in multi_seed_source
        and "Strategy Compare continuous score起始覆蓋不足" in strategy_training_source
        and "Strategy Compare continuous score結束覆蓋不足" in strategy_training_source
        and "param_evaluation_mode=param_evaluation_mode" in multi_seed_source
        and "_strategy_only_baseline_context_available(" in multi_seed_source
        and '"param_evaluation_mode": str(' in strategy_reuse_source
        and "expected_param_evaluation_mode=str(param_evaluation_mode)" in strategy_replay_source
        and "expected_param_evaluation_mode=str(param_evaluation_mode)" in strategy_engine_source,
    )

    check_true(
        "robustness_attribution_progress_separates_missing_observation_from_attribution_rebuild",
        "expected_model_units =" in multi_seed_source
        and "scientific observation pending=" in multi_seed_source
        and "attribution rebuild=" in multi_seed_source
        and "scientific_completed - attribution_ready" not in multi_seed_source,
    )

    check_true(
        "robustness_report_surfaces_run_pinned_seed_and_optimizer_budget_identity",
        '("Benchmark ID", str(contract.get("benchmark_id") or "-"))' in multi_seed_source
        and '("Strategy trials/fold", str(contract.get("strategy_trials_per_fold") or "-"))' in multi_seed_source
        and '("Seed generator", str(contract.get("seed_generator_seed") or "-"))' in multi_seed_source
        and "非same-seed配對勝率" in multi_seed_source,
    )

    oos_robustness = get_strategy_multi_seed_robustness_settings("extending_window_oos")
    rolling_robustness = get_strategy_multi_seed_robustness_settings("extending_window_rolling")
    oos_cache_root = str(oos_robustness.checkpoint_cache_root or "").strip()
    rolling_cache_root = str(rolling_robustness.checkpoint_cache_root or "").strip()
    check_true(
        "current_oos_and_rolling_robustness_share_fitting_identity_checkpoint_cache",
        bool(oos_cache_root)
                and oos_cache_root == rolling_cache_root
                and not Path(oos_cache_root).is_absolute()
                and "multi_seed_robustness" not in oos_cache_root
                and "STRATEGY_COMPARE_FITTING_CHECKPOINT_CACHE_ROOT" in strategy_compare_config_source
                and "--checkpoint-cache-root" in strategy_training_source
                and "fitting_identity_checkpoint_reuse" in point_in_time_source
                and "source_score_reused" in point_in_time_source
                and "fold_training_identity" in point_in_time_source,
    )

    check_true(
        "strategy_compare_training_progress_uses_shared_multiline_board_and_live_fold_epoch_ssot",
        "FixedProgressBlock" in multi_seed_source
        and "render_training_unit_progress" in multi_seed_source
        and "read_trainer_epoch_progress" in multi_seed_source
        and "read_trainer_pit_progress" in multi_seed_source
        and "_pit_saved_fold_progress" not in multi_seed_source
        and "read_trainer_pit_progress" in strategy_app_service_source
        and 'COMPACT_CONSOLE_ENV: "0"' in strategy_training_source
        and '"PYTHONUNBUFFERED": "1"' in strategy_training_source
        and '"BREAKOUT_QUALITY_EPOCH_PROGRESS_MARKERS": "1"' in strategy_training_source
        and '"BREAKOUT_QUALITY_PIT_PROGRESS_MARKERS": "1"' in strategy_training_source
        and "def read_trainer_epoch_progress(" in training_progress_source
        and "def read_trainer_pit_progress(" in training_progress_source
        and "_EPOCH_PROGRESS_MARKER_RE" in training_progress_source
        and "_PIT_PROGRESS_MARKER_RE" in training_progress_source
        and "set_trainer_pit_progress_context" in point_in_time_source
        and "emit_trainer_pit_progress_marker" in continuous_ranker_source
        and "__BQ_EPOCH_PROGRESS__" in continuous_ranker_source
        and "_emit_epoch_progress_marker(\"select\"" in continuous_ranker_source
        and "_emit_epoch_progress_marker(\"refit\"" in continuous_ranker_source
        and "active fold" in training_progress_source
        and "epoch pending" in training_progress_source
        and "epoch {phase}" in training_progress_source
        and "FixedProgressBlock" in strategy_app_service_source
        and "render_training_unit_progress(" in strategy_app_service_source,
    )

    check_true(
        "multi_seed_robustness_reuses_strategy_compare_canonical_report_renderers_and_keeps_seed_extensions_separate",
        "render_strategy_aggregate_report" in multi_seed_source
                and "common_strategy_report" in multi_seed_source
                and "RoMD完整統計" in multi_seed_source
                and "設定中的同seed contrasts" in multi_seed_source
                and "歷年報酬跨seed完整統計" in multi_seed_source
                and "_upgrade_derived_report_summary" in multi_seed_source
                and "report_refreshed_at_utc" in multi_seed_source,
    )

    check_true(
        "strategy_compare_and_robustness_share_execution_plan_renderer_and_status_color_ssot",
        "def render_strategy_execution_plan_surface(" in comparison_source
                and "return render_strategy_execution_plan_surface(" in comparison_source
                and "render_strategy_execution_plan_surface(" in multi_seed_source
                and "def _compact_execution_plan_item(" in comparison_source
                and "def _compact_execution_plan_description(" in comparison_source
                and "_EXECUTION_PLAN_ITEM_WIDTH = 36" in comparison_source
                and "_EXECUTION_PLAN_DESCRIPTION_WIDTH = 44" in comparison_source
                and "action_colors =" not in multi_seed_source
                and "def signal_for_workflow_status(" in report_style_source
                and "def styled_workflow_status(" in report_style_source
                and "styled_workflow_status" in comparison_source,
    )


    check_true(
        "strategy_reports_share_metric_registry_and_project_wide_color_semantics",
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
        _full_horizon_path_lookup_cached,
        _pure_mfe_diagnostic_profile,
        paired_trade_r_conversion_diagnostic,
    )
    from config.breakout_quality import (
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
    )
    from config.strategy_compare import STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE
    from filters.breakout_quality.continuous_target import (
        DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )

    diagnostic_owner = _pure_mfe_diagnostic_profile()
    diagnostic_owner_profile = get_breakout_quality_experiment_profile(diagnostic_owner)
    conditional_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    check_true(
        "upside_path_diagnostic_owner_is_explicit_when_research_profiles_share_pure_mfe_target",
        diagnostic_owner == STRATEGY_COMPARE_UPSIDE_REALIZATION_PATH_PROFILE
        and diagnostic_owner != DAILY_UNIVERSAL_CONDITIONAL_MFE_SAFETY_FULL_LIST_NDCG_PAIRWISE_PROFILE
        and str(diagnostic_owner_profile.continuous_target_id or "")
        == str(conditional_profile.continuous_target_id or "")
        == DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
    )
    from filters.breakout_quality.trade_attribution import (
        build_upside_realization_attribution,
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
    check_true(
        "rce_uses_same_exclusive_completed_trade_universe_for_target_and_realized_edges",
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
    check_true(
        "rce_excludes_missing_target_trade_from_both_edges_without_100pct_coverage_gate",
        partial_rce.get("complete_target_coverage") is False
                and int(partial_rce.get("active_only_count") or 0) == 2
                and int(partial_rce.get("active_only_target_covered_count") or 0) == 1
                and abs(float(partial_rce.get("target_coverage_rate")) - (2.0 / 3.0)) < 1e-12
                and abs(float(partial_rce.get("paired_target_selection_edge_r")) - 1.0) < 1e-12
                and abs(float(partial_rce.get("paired_realized_selection_edge_r")) - 2.0) < 1e-12
                and abs(float(partial_rce.get("r_conversion_efficiency")) - 2.0) < 1e-12,
    )



    # Canonical sanitized OHLCV keeps trading dates in the DatetimeIndex, not in
    # a physical ``Date`` column.  First-passage timing must consume that same index.
    from filters.breakout_quality.daily_ranker_data import LazyDailyFeatureBank

    canonical_index_frame = pd.DataFrame(
        {
            "Open": [10.0, 10.0, 10.0, 10.0],
            "High": [10.1, 11.1, 12.1, 13.1],
            "Low": [9.9, 9.8, 9.7, 9.6],
            "Close": [10.0, 10.5, 11.0, 12.0],
            "Volume": [1000.0, 1000.0, 1000.0, 1000.0],
        },
        index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )
    canonical_index_bank = LazyDailyFeatureBank(
        frames=(canonical_index_frame,),
        benchmark=canonical_index_frame,
        ticker_ids=np.asarray([0], dtype=np.int32),
        source_positions=np.asarray([0], dtype=np.int32),
        benchmark_positions=np.asarray([0], dtype=np.int32),
        policy=SimpleNamespace(feature_window_bars=1),
    )
    canonical_index_passage = canonical_index_bank.future_first_passage(
        np.asarray([0], dtype=np.int64),
        horizon_bars=3,
        return_thresholds=(0.1, 0.2),
    )
    check_true(
        "daily_first_passage_uses_canonical_datetime_index_without_date_column",
        "Date" not in canonical_index_frame.columns
        and int(canonical_index_passage[0.1][0][0]) == 1
        and str(canonical_index_passage[0.1][1][0]) == "2020-01-02"
        and int(canonical_index_passage[0.2][0][0]) == 2
        and str(canonical_index_passage[0.2][1][0]) == "2020-01-03",
    )

    pure_mfe_groups = pd.DataFrame([
        {
            "ticker": "AAA", "group_index": 0, "date": "2020-01-01", "target_adverse_r": 0.4,
            "target_opportunity_bar": 3, "target_first_risk_breach_bar": -1,
            "target_opportunity_date": "2020-01-04", "target_first_risk_breach_date": pd.NaT,
        },
        {
            "ticker": "BBB", "group_index": 1, "date": "2020-01-02", "target_adverse_r": float("nan"),
            "target_opportunity_bar": -1, "target_first_risk_breach_bar": -1,
            "target_opportunity_date": pd.NaT, "target_first_risk_breach_date": pd.NaT,
        },
    ])
    def synthetic_first_passage(group_ids, *, horizon_bars, return_thresholds):
        assert list(group_ids) == [0, 1]
        assert int(horizon_bars) == 3
        result = {}
        for threshold in return_thresholds:
            if abs(float(threshold) - 0.1) < 1e-12:
                result[threshold] = (
                    np.asarray([1, -1], dtype=np.int16),
                    np.asarray(["2020-01-02", "NaT"], dtype="datetime64[D]"),
                )
            elif abs(float(threshold) - 0.2) < 1e-12:
                result[threshold] = (
                    np.asarray([2, -1], dtype=np.int16),
                    np.asarray(["2020-01-03", "NaT"], dtype="datetime64[D]"),
                )
            else:
                result[threshold] = (
                    np.asarray([-1, -1], dtype=np.int16),
                    np.asarray(["NaT", "NaT"], dtype="datetime64[D]"),
                )
        return result

    pure_mfe_bundle = SimpleNamespace(
        group_table=pure_mfe_groups,
        raw_target=np.asarray([2.5, np.nan], dtype=np.float32),
        target_valid=np.asarray([True, False], dtype=bool),
        target_manifest={
            "target_contract": {
                "target_id": "daily_full_horizon_pure_mfe_r_v1",
                "risk_budget_return": 0.1,
                "horizon_bars": 3,
            }
        },
        feature_bank=SimpleNamespace(future_first_passage=synthetic_first_passage),
    )
    with patch(
        "filters.breakout_quality.strategy_compare_diagnostics.load_profile_continuous_ranker_data",
        return_value=pure_mfe_bundle,
    ) as path_loader:
        path_lookup = _full_horizon_path_lookup_cached.__wrapped__(
            ".", "breakout_quality_v1", "inception_time_predicted_upside_context_v1"
        )
    path_loader_kwargs = dict(path_loader.call_args.kwargs)
    check_true(
        "upside_path_lookup_uses_canonical_pure_mfe_architecture_not_active_arm_architecture",
        path_loader_kwargs.get("experiment_profile") == diagnostic_owner
        and path_loader_kwargs.get("model_architecture") == "inception_time_v1",
    )
    check_true(
        "upside_path_lookup_consumes_canonical_daily_provider_without_nonexistent_persisted_component_bundle",
        len(path_lookup) == 2
        and bool(path_lookup.loc[0, "path_target_available"])
        and not bool(path_lookup.loc[1, "path_target_available"])
        and abs(float(path_lookup.loc[0, "full_horizon_mfe_r"]) - 2.5) < 1e-12
        and abs(float(path_lookup.loc[0, "full_horizon_adverse_to_peak_r"]) - 0.4) < 1e-6
        and int(path_lookup.loc[0, "full_horizon_opportunity_bar"]) == 3
        and int(path_lookup.loc[0, "full_horizon_first_upside_1r_bar"]) == 1
        and str(path_lookup.loc[0, "full_horizon_first_upside_2r_date"]) == "2020-01-03"
        and int(path_lookup.loc[0, "full_horizon_first_upside_3r_bar"]) == -1,
    )

    stop_trades = pd.DataFrame([
        {"Date": "2020-02-03", "Ticker": "UP", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-02-02", "成交價": 10.0, "停損價": 9.0},
        {"Date": "2020-02-05", "Ticker": "UP", "Type": "全倉結算(停損)", "成交價": 9.0, "停損價": 9.0, "R_Multiple": -1.0, "該筆總損益": -100.0},
        {"Date": "2020-02-03", "Ticker": "OK", "Type": "買進 (test)", "進場類型": "normal", "買訊日": "2020-02-02", "成交價": 10.0, "停損價": 9.0},
        {"Date": "2020-02-10", "Ticker": "OK", "Type": "全倉結算(指標)", "成交價": 12.0, "R_Multiple": 2.0, "該筆總損益": 200.0},
    ])
    stop_paths = pd.DataFrame([
        {
            "ticker": "UP", "trade_date": "2020-02-03", "signal_date": "2020-02-02",
            "score_event_date": "2020-02-02", "path_target_available": True,
            "full_horizon_mfe_r": 3.0, "full_horizon_adverse_to_peak_r": 1.2,
            "full_horizon_opportunity_date": "2020-02-07",
            "full_horizon_first_risk_breach_date": "2020-02-05",
            "full_horizon_opportunity_bar": 5, "full_horizon_first_risk_breach_bar": 3,
            "full_horizon_first_upside_1r_bar": 1, "full_horizon_first_upside_1r_date": "2020-02-03",
            "full_horizon_first_upside_2r_bar": 2, "full_horizon_first_upside_2r_date": "2020-02-04",
            "full_horizon_first_upside_3r_bar": 5, "full_horizon_first_upside_3r_date": "2020-02-07",
        },
        {
            "ticker": "OK", "trade_date": "2020-02-03", "signal_date": "2020-02-02",
            "score_event_date": "2020-02-02", "path_target_available": True,
            "full_horizon_mfe_r": 2.5, "full_horizon_adverse_to_peak_r": 0.3,
            "full_horizon_opportunity_date": "2020-02-09",
            "full_horizon_first_risk_breach_date": "",
            "full_horizon_opportunity_bar": 6, "full_horizon_first_risk_breach_bar": -1,
            "full_horizon_first_upside_1r_bar": 1, "full_horizon_first_upside_1r_date": "2020-02-03",
            "full_horizon_first_upside_2r_bar": 3, "full_horizon_first_upside_2r_date": "2020-02-05",
            "full_horizon_first_upside_3r_bar": -1, "full_horizon_first_upside_3r_date": "",
        },
    ])
    realization = build_upside_realization_attribution(
        trade_history=stop_trades,
        selected_path_diagnostics=stop_paths,
        upside_r_thresholds=(1.0, 2.0, 3.0),
        adverse_bucket_edges_r=(0.5, 1.0),
        scenario="synthetic",
    )
    realization_summary = dict(realization.get("summary") or {})
    check_true(
        "upside_realization_counts_actual_stop_before_later_full_horizon_peak_on_canonical_completed_trades",
        int(realization_summary.get("completed_trade_count") or 0) == 2
        and int(realization_summary.get("path_target_covered_count") or 0) == 2
        and int(realization_summary.get("actual_stop_out_count") or 0) == 1
        and int(realization_summary.get("stop_before_later_peak_count") or 0) == 1
        and int((realization_summary.get("thresholds") or {}).get("2R", {}).get("stop_before_later_peak_count") or 0) == 1
        and abs(float((realization_summary.get("thresholds") or {}).get("3R", {}).get("stop_before_later_peak_rate")) - 1.0) < 1e-12
        and int((realization_summary.get("thresholds") or {}).get("2R", {}).get("actual_stop_before_first_upside_count") or 0) == 0
        and int((realization_summary.get("thresholds") or {}).get("3R", {}).get("actual_stop_before_first_upside_count") or 0) == 1
        and int((realization_summary.get("thresholds") or {}).get("3R", {}).get("actual_initial_stop_before_first_upside_count") or 0) == 1
        and int((realization_summary.get("thresholds") or {}).get("3R", {}).get("canonical_risk_before_first_upside_count") or 0) == 1
        and int((realization_summary.get("thresholds") or {}).get("2R", {}).get("canonical_upside_before_risk_count") or 0) == 2,
    )
    check_true(
        "strategy_diagnostics_has_independent_stale_contract_and_reuse_backfill_without_engine_schema_invalidation",
        "STRATEGY_DIAGNOSTICS_SCHEMA_VERSION" in diagnostics_source
        and "diagnostics_contract_fingerprint" in diagnostics_source
        and "backfill_pair_upside_realization_diagnostic" in comparison_source
        and "backfill_pair_upside_realization_diagnostic" in diagnostics_source
        and "pair_upside_realization_refresh_required" in comparison_source
        and '"REFRESH"' in comparison_source
        and '"Diagnostics | Upside Realization"' in comparison_source
        and 'section_contract("strategy.standard_sop", "upside_survival").display_title' in comparison_source
        and "render_upside_survival_summary_table" in comparison_source
        and "STRATEGY_COMPARE_UPSIDE_REALIZATION_R_THRESHOLDS" in strategy_compare_config_source,
    )

    from filters.breakout_quality.strategy_compare_diagnostics import (
        render_first_passage_summary_table,
        render_upside_realization_summary_table,
        render_upside_survival_summary_table,
    )
    unavailable_text = render_upside_survival_summary_table({
        "upside_realization": [],
        "upside_realization_unavailable": [{
            "arm_id": "C59",
            "name": "Synthetic",
            "status": "UNAVAILABLE",
            "reason": "ValueError: synthetic reason",
        }],
    })
    check_true(
        "upside_realization_unavailable_reason_is_visible_on_main_report",
        "C59 Synthetic" in unavailable_text
        and "ValueError: synthetic reason" in unavailable_text,
    )

    first_passage_text = render_first_passage_summary_table({
        "upside_realization": [{
            "arm_id": "C59", "name": "Synthetic",
            "thresholds": realization_summary.get("thresholds"),
        }],
        "upside_realization_contract": {"upside_r_thresholds": [1.0, 2.0, 3.0]},
    })
    check_true(
        "first_passage_main_report_distinguishes_peak_before_stop_from_exact_threshold_timing",
        "First-Passage Realization" in first_passage_text
        and "+2R" in first_passage_text
        and "0/2 (0.00%)" in first_passage_text
        and "+3R" in first_passage_text
        and "1/1 (100.00%)" in first_passage_text,
    )

    compact_diagnostics = {
        "upside_realization": [
            {
                "arm_id": "C59", "name": "Synthetic E",
                "full_horizon_mfe_mean_r": 1.20,
                "full_horizon_adverse_to_peak_mean_r": 0.30,
                "realized_mean_r": 1.10,
                "thresholds": {
                    "1R": {"actual_initial_stop_before_first_upside_rate": 0.10},
                    "2R": {"actual_initial_stop_before_first_upside_rate": 0.08},
                    "3R": {"actual_initial_stop_before_first_upside_rate": 0.05},
                },
            },
            {
                "arm_id": "C60", "name": "Synthetic KM",
                "full_horizon_mfe_mean_r": 0.90,
                "full_horizon_adverse_to_peak_mean_r": 0.20,
                "realized_mean_r": 0.80,
                "thresholds": {
                    "1R": {"actual_initial_stop_before_first_upside_rate": 0.20},
                    "2R": {"actual_initial_stop_before_first_upside_rate": 0.22},
                    "3R": {"actual_initial_stop_before_first_upside_rate": 0.15},
                },
            },
        ],
        "upside_realization_contract": {"upside_r_thresholds": [1.0, 2.0, 3.0]},
    }
    compact_text = render_upside_survival_summary_table(compact_diagnostics, target="plain")
    check_true(
        "upside_survival_main_report_is_compact_and_retains_decision_metrics",
        "+1R前初始Stop" in compact_text
        and "+2R前初始Stop" in compact_text
        and "+3R前初始Stop" in compact_text
        and "Full-MFE" in compact_text
        and "Adverse" in compact_text
        and "Realized EV" in compact_text
        and "Path Coverage" not in compact_text
        and "Risk≤首次達標" not in compact_text
        and "其中拉高Stop" not in compact_text,
    )

    compact_markdown = render_upside_survival_summary_table(
        compact_diagnostics, target="markdown"
    )
    check_true(
        "upside_survival_main_report_uses_shared_best_worst_color_semantics",
        '#188038' in compact_markdown
        and '#C62828' in compact_markdown
        and '<span style="color:#188038;">8.00%</span>' in compact_markdown
        and '<span style="color:#C62828;">22.00%</span>' in compact_markdown
        and '<span style="color:#C62828;">0.30R</span>' in compact_markdown
        and '<span style="color:#188038;">0.20R</span>' in compact_markdown,
    )

    check_true(
        "strategy_diagnostics_reuse_canonical_producers_without_target_formula_duplication",
        '"path_components_source": "canonical daily profile sample provider"' in diagnostics_source
                and '"diagnostic_target_formula_reimplementation": False' in diagnostics_source
                and '"canonical_provider_may_materialize_from_source_ohlcv": True' in diagnostics_source
                and '"first_passage_timing_source": "canonical_daily_ohlcv_used_by_pure_mfe_sample_provider"' in diagnostics_source
                and "future_first_passage" in diagnostics_source
                and "load_validated_continuous_target_component_arrays" not in diagnostics_source
                and "selection_diagnostics" in diagnostics_source,
    )

    from dataclasses import replace
    from config.strategy_compare import get_strategy_comparison_settings
    from filters.breakout_quality.strategy_comparison import render_safety_gate_sensitivity_table

    # C71-C73 are historical-only now.  Keep testing the reusable sensitivity renderer
    # with a deliberate historical fixture instead of requiring retired arms in current.
    current_settings = get_strategy_comparison_settings("extending_window_oos")
    historical_gate_ids = {"C58", "C71", "C72", "C73"}
    sensitivity_settings = replace(
        current_settings,
        arms={
            arm_id: replace(
                arm,
                enabled=arm_id in historical_gate_ids,
                param_source=("min_rolling" if arm_id == "C58" else arm.param_source),
            )
            for arm_id, arm in current_settings.arms.items()
        },
        contrasts={
            contrast_id: replace(contrast, enabled=False)
            for contrast_id, contrast in current_settings.contrasts.items()
        },
    )
    sensitivity_scenarios = {
        arm_id: {"avg_exposure_pct": exposure, "return_over_max_drawdown": romd}
        for arm_id, exposure, romd in (("C71", 62.0, 6.0), ("C72", 64.0, 7.0), ("C73", 61.0, 6.5))
    }
    sensitivity_diagnostics = {
        "mfe_safety_geometry": {
            "status": "AVAILABLE",
            "arms": {
                arm_id: {
                    "high_mfe_high_safety_pct": hmhs,
                    "high_mfe_low_safety_pct": hmls,
                    "high_mfe_total_pct": hm,
                    "high_safety_total_pct": hs,
                }
                for arm_id, hmhs, hmls, hm, hs in (
                    ("C71", 25.0, 25.0, 50.0, 49.0),
                    ("C72", 28.0, 20.0, 48.0, 56.0),
                    ("C73", 29.0, 15.0, 44.0, 63.0),
                )
            },
        },
        "upside_realization": [
            {"arm_id": arm_id, "full_horizon_mfe_mean_r": mfe, "full_horizon_adverse_to_peak_mean_r": adverse, "realized_mean_r": ev}
            for arm_id, mfe, adverse, ev in (("C71", 1.4, 0.34, 0.54), ("C72", 1.3, 0.28, 0.62), ("C73", 1.1, 0.24, 0.58))
        ],
    }
    sensitivity_text = render_safety_gate_sensitivity_table(
        sensitivity_scenarios, sensitivity_diagnostics, settings=sensitivity_settings
    )
    check_true(
        "raw_safety_gate_sensitivity_table_renders_only_p50_p60_p70_curve_and_key_metrics",
        all(token in sensitivity_text for token in ("C71", "C72", "C73", ">=0.50", ">=0.60", ">=0.70", "HM/HS", "HM/LS", "Full-MFE", "Adverse", "Realized EV", "平均曝險", "RoMD"))
        and "C70" not in sensitivity_text
        and "C64/C66" in sensitivity_text,
    )

    from filters.breakout_quality.strategy_compare_reuse import _pair_group_id
    from filters.breakout_quality.strategy_compare_runtime import _execution_pairs, _arm_runtime_spec
    from filters.breakout_quality.strategy_comparison import _scenario_payloads

    sensitivity_pairs = [
        row for row in _execution_pairs(sensitivity_settings)
        if row[3].arm_id in {"C71", "C72", "C73"}
    ]
    sensitivity_group_ids = {
        on_arm.arm_id: _pair_group_id(
            param_source=param_source,
            rule_policy=rule_policy,
            dl_id=str(on_arm.dl_id or ""),
            dl_runtime_mode=str(on_arm.dl_runtime_mode or ""),
            arm_id=on_arm.arm_id,
        )
        for param_source, rule_policy, _off_arm, on_arm in sensitivity_pairs
    }
    check_true(
        "raw_safety_gate_sensitivity_pair_storage_keys_are_arm_unique",
        set(sensitivity_group_ids) == {"C71", "C72", "C73"}
        and len(set(sensitivity_group_ids.values())) == 3
        and all(arm_id in group_id for arm_id, group_id in sensitivity_group_ids.items()),
    )

    synthetic_pair_payloads = {}
    synthetic_direct_r = {}
    for param_source, rule_policy, off_arm, on_arm in sensitivity_pairs:
        group_id = sensitivity_group_ids[on_arm.arm_id]
        active_key = _arm_runtime_spec(on_arm)["active_key"]
        synthetic_pair_payloads[group_id] = {
            "arm_contract": (param_source, rule_policy, off_arm, on_arm),
            "payload": {
                "no_filter": {"total_return_pct": 1.0},
                active_key: {
                    "total_return_pct": float({"C71": 1, "C72": 2, "C73": 3}[on_arm.arm_id]),
                    "avg_exposure_pct": 60.0,
                },
            },
        }
        synthetic_direct_r[group_id] = 0.0
    synthetic_scenarios = _scenario_payloads(
        synthetic_pair_payloads, synthetic_direct_r, settings=sensitivity_settings
    )
    check_true(
        "raw_safety_gate_sensitivity_scenario_materialization_keeps_all_three_arms",
        all(arm_id in synthetic_scenarios for arm_id in ("C71", "C72", "C73"))
        and synthetic_scenarios["C71"]["total_return_pct"] == 1.0
        and synthetic_scenarios["C72"]["total_return_pct"] == 2.0
        and synthetic_scenarios["C73"]["total_return_pct"] == 3.0,
    )

    summary["strategy_output_contracts"] = [row[0] for row in contract_rows]
    return results, summary
