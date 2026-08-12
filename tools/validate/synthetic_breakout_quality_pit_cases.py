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
    add_check,
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

def validate_breakout_quality_point_in_time_score_builder_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_BUILDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

    from config import breakout_quality as workflow_config
    from config.breakout_quality import get_breakout_quality_workflow_settings
    from tools.audit.breakout_quality.point_in_time_scores import (
        _direction_summary,
        _orderable_coverage,
        _render_markdown as render_point_in_time_markdown,
        render_compact_console_summary as render_point_in_time_compact_console,
        render_console_summary as render_point_in_time_console,
    )
    from tools.filters.breakout_quality.build_point_in_time_scores import (
        REQUIRED_SCORE_COLUMNS,
        _build_fold_periods,
        _stable_fold_id,
        _combined_validation,
        _fold_group_ids,
        _fold_training_contract_is_compatible,
        _migrate_compatible_legacy_fold,
        _rescore_daily_fold_from_compatible_checkpoint,
        _resolve_score_start,
        _validate_score_frame,
        parse_args as parse_point_in_time_args,
    )
    from filters.breakout_quality.continuous_ranker_data import _validate_group_consistency
    from filters.breakout_quality.ranker_sample_contract import (
        build_score_eligibility_contract,
        resolve_forward_oos_score_group_ids,
        resolve_forward_oos_target_evaluable_group_ids,
    )
    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME, LABEL_PASS

    consistent_terminal_events = pd.DataFrame(
        [
            {
                "ticker": "2330",
                "date": "2026-03-02",
                "group_index": 0,
                "label_eval_end_date": None,
            },
            {
                "ticker": "2330",
                "date": "2026-03-02",
                "group_index": 0,
                "label_eval_end_date": None,
            },
            {
                "ticker": "2317",
                "date": "2020-01-02",
                "group_index": 1,
                "label_eval_end_date": "2020-03-02",
            },
        ]
    )
    all_missing_label_end_accepted = True
    try:
        _validate_group_consistency(consistent_terminal_events)
    except ValueError:
        all_missing_label_end_accepted = False
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_group_consistency_accepts_all_missing_terminal_label_end",
        True,
        all_missing_label_end_accepted,
    )

    event_profile = get_breakout_quality_experiment_profile(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    forward_bundle = SimpleNamespace(
        group_table=pd.DataFrame([
            {"ticker": "A", "date": "2021-01-04", "group_index": 0, "label": 1},
            {"ticker": "B", "date": "2021-01-04", "group_index": 1, "label": 0},
            {"ticker": "C", "date": "2021-01-05", "group_index": 2, "label": -1},
            {"ticker": "D", "date": "2020-12-31", "group_index": 3, "label": 1},
        ]),
        target_valid=np.array([True, False, True, True], dtype=bool),
        outer_policy={
            "oos_start_date": "2021-01-01",
            "effective_oos_end_date": "2021-12-31",
        },
        profile=event_profile,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "forward_oos_runtime_score_universe_does_not_require_future_target_completion",
        ([0, 1, 2], [0]),
        (
            resolve_forward_oos_score_group_ids(forward_bundle).tolist(),
            resolve_forward_oos_target_evaluable_group_ids(forward_bundle).tolist(),
        ),
    )

    mixed_label_end_rejected = False
    inconsistent_terminal_events = consistent_terminal_events.copy()
    inconsistent_terminal_events.loc[1, "label_eval_end_date"] = "2026-04-30"
    try:
        _validate_group_consistency(inconsistent_terminal_events)
    except ValueError as exc:
        mixed_label_end_rejected = "invalid_groups=1" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_group_consistency_rejects_mixed_missing_and_completed_label_end",
        True,
        mixed_label_end_rejected,
    )

    settings = get_breakout_quality_workflow_settings()
    with patch.object(
        workflow_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE,
    ):
        binary_settings = workflow_config.get_breakout_quality_workflow_settings()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "workflow_binary_profile_resolves_classification_and_hard_filter",
        (
            "binary_classification",
            None,
            "hard-filter",
            "canonical_runtime",
            "original",
        ),
        (
            binary_settings.training_objective,
            binary_settings.continuous_target_id,
            binary_settings.strategy_comparison_mode,
            binary_settings.strategy_score_source,
            binary_settings.strategy_buy_sort,
        ),
    )
    with patch.object(
        workflow_config,
        "BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE",
        STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE,
    ):
        continuous_settings = workflow_config.get_breakout_quality_workflow_settings()
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "workflow_continuous_profile_resolves_point_in_time_score_ranking",
        (
            "daily_percentile_regression",
            "strategy_aligned_opportunity_no_time_r_v1",
            "score-ranking",
            "selection_point_in_time",
            "breakout_quality_score_desc",
        ),
        (
            continuous_settings.training_objective,
            continuous_settings.continuous_target_id,
            continuous_settings.strategy_comparison_mode,
            continuous_settings.strategy_score_source,
            continuous_settings.strategy_buy_sort,
        ),
    )
    daily_settings = workflow_config.get_breakout_quality_workflow_settings(
        experiment_profile=DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_universal_profile_enables_model_layer_pit_without_changing_strategy_identity",
        (
            "daily_pairwise_ranking",
            "daily_eligible_stock_days",
            True,
            settings.experiment_profile,
        ),
        (
            daily_settings.training_objective,
            daily_settings.training_sample_scope,
            daily_settings.supports_point_in_time_scores,
            workflow_config.BREAKOUT_QUALITY_WORKFLOW_EXPERIMENT_PROFILE,
        ),
    )
    parsed = parse_point_in_time_args([])
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_cli_defaults_follow_current_workflow_config",
        (
            settings.filter_id,
            settings.model_architecture,
            settings.experiment_profile,
            settings.seed,
            settings.point_in_time_score_start_date,
            settings.point_in_time_fold_months,
            settings.point_in_time_inner_validation_months,
        ),
        (
            parsed.filter_id,
            parsed.model_architecture,
            parsed.experiment_profile,
            parsed.seed,
            parsed.score_start_date,
            parsed.fold_months,
            parsed.inner_validation_months,
        ),
    )

    periods = _build_fold_periods(
        pd.Timestamp("2019-12-31"),
        pd.Timestamp("2020-03-15"),
        fold_months=1,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_fold_periods_are_contiguous_and_calendar_month_based",
        [
            ("2019-12-31", "2019-12-31"),
            ("2020-01-01", "2020-01-31"),
            ("2020-02-01", "2020-02-29"),
            ("2020-03-01", "2020-03-15"),
        ],
        [
            (str(item["score_start"].date()), str(item["score_end"].date()))
            for item in periods
        ],
    )

    original_annual_periods = _build_fold_periods(
        pd.Timestamp("2014-01-01"),
        pd.Timestamp("2015-12-31"),
        fold_months=12,
    )
    earlier_periods = _build_fold_periods(
        pd.Timestamp("2007-06-01"),
        pd.Timestamp("2015-12-31"),
        fold_months=12,
    )
    original_fold = next(
        item for item in original_annual_periods
        if item["score_start"] == pd.Timestamp("2014-01-01")
    )
    extended_fold = next(
        item for item in earlier_periods
        if item["score_start"] == pd.Timestamp("2014-01-01")
    )
    auto_group_dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    auto_bundle = SimpleNamespace(
        group_table=pd.DataFrame({
            "date": auto_group_dates,
            "label_eval_end_date": auto_group_dates,
            "label": np.full(len(auto_group_dates), LABEL_PASS, dtype=np.int64),
        }),
        target_valid=np.ones(len(auto_group_dates), dtype=bool),
        profile=SimpleNamespace(
            training_label_scope=TRAINING_LABEL_SCOPE_PASS_ONLY,
            training_sample_scope=TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
        ),
    )
    auto_settings = SimpleNamespace(
        point_in_time_min_train_groups=2,
        point_in_time_min_validation_groups=2,
        point_in_time_min_score_groups=1,
    )
    resolved_auto_start, auto_diagnostics = _resolve_score_start(
        "auto",
        bundle=auto_bundle,
        settings=auto_settings,
        selection_start=pd.Timestamp("2020-01-01"),
        score_end=pd.Timestamp("2020-08-31"),
        fold_months=1,
        validation_months=2,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_auto_start_selects_earliest_month_meeting_real_split_counts",
        ("2020-05-01", 5, {"inner_train": 2, "validation": 2, "score": 1}),
        (
            str(resolved_auto_start.date()),
            auto_diagnostics["candidate_months_checked"],
            auto_diagnostics["first_fold_group_counts"],
        ),
    )

    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_fold_identity_is_date_stable_when_history_is_extended",
        (
            "fold_20140101_20141231",
            "fold_20140101_20141231",
            "fold_20140101_20141231",
        ),
        (
            original_fold["fold_id"],
            extended_fold["fold_id"],
            _stable_fold_id(pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")),
        ),
    )

    with tempfile.TemporaryDirectory() as tmp:
        pit_root = Path(tmp) / "point_in_time"
        legacy_dir = pit_root / "folds" / "fold_000"
        target_dir = pit_root / "folds" / "fold_20140101_20141231"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        model_path = legacy_dir / DEFAULT_MODEL_FILENAME
        score_path = legacy_dir / "scores.csv"
        pd.DataFrame(
            [{
                "ticker": "2330",
                "date": "2014-06-30",
                "group_index": 7,
                "breakout_quality_score": 0.75,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-31",
            }]
        ).to_csv(score_path, index=False, encoding="utf-8-sig")
        common_contract = {
            "filter_id": "breakout_quality_v1",
            "model_architecture": "inception_time_v1",
            "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
            "continuous_target_id": "strategy_aligned_opportunity_no_time_r_v1",
            "training_label_scope": "pass_only",
            "seed": 42,
            "planned_periods": {
                "score_start": "2014-01-01",
                "score_end": "2014-12-31",
            },
            "observed_periods": {},
            "model_information_cutoff": "2013-12-31",
            "group_counts": {},
            "event_row_counts": {},
            "model_spec": {},
            "experiment_settings": {},
            "training_settings": {},
            "source_contract": {},
            "lookahead_contract": {},
        }
        import torch

        torch.save(
            {
                "model_state_dict": {"synthetic_weight": torch.tensor([1.0])},
                "fold_contract": {
                    **common_contract,
                    "schema_version": 1,
                    "fold_id": "fold_000",
                },
            },
            model_path,
        )
        legacy_manifest = {
            **common_contract,
            "schema_version": 1,
            "fold_id": "fold_000",
            "contract_fingerprint": "legacy-fingerprint",
            "artifacts": {
                "checkpoint": build_file_manifest(model_path),
                "scores": build_file_manifest(score_path),
            },
        }
        (legacy_dir / "manifest.json").write_text(
            json.dumps(legacy_manifest), encoding="utf-8"
        )
        target_contract = {
            **common_contract,
            "schema_version": 2,
            "fold_id": "fold_20140101_20141231",
        }
        migrated = _migrate_compatible_legacy_fold(
            point_in_time_dir=pit_root,
            target_fold_dir=target_dir,
            expected_fingerprint="stable-fingerprint",
            fold_contract=target_contract,
            torch_module=torch,
        )
        migrated_frame, migrated_manifest = migrated or (pd.DataFrame(), {})
        migrated_checkpoint = (
            torch.load(
                target_dir / DEFAULT_MODEL_FILENAME,
                map_location="cpu",
                weights_only=True,
            )
            if migrated is not None
            else {}
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "compatible_legacy_fold_is_hash_checked_and_migrated_to_stable_id",
        (
            True,
            {"fold_20140101_20141231"},
            "fold_20140101_20141231",
            "fold_000",
            "stable-fingerprint",
        ),
        (
            migrated is not None,
            set(migrated_frame.get("fold_id", pd.Series(dtype=str)).astype(str)),
            dict(migrated_checkpoint.get("fold_contract") or {}).get("fold_id"),
            dict(migrated_manifest.get("migration") or {}).get("source_fold_id"),
            migrated_manifest.get("contract_fingerprint"),
        ),
    )

    group_table = pd.DataFrame(
        [
            {
                "ticker": "A",
                "date": "2010-01-01",
                "group_index": 0,
                "label": 1,
                "label_eval_end_date": "2010-02-01",
            },
            {
                "ticker": "B",
                "date": "2011-12-01",
                "group_index": 1,
                "label": 1,
                "label_eval_end_date": "2012-01-10",
            },
            {
                "ticker": "C",
                "date": "2012-01-02",
                "group_index": 2,
                "label": 1,
                "label_eval_end_date": "2012-02-15",
            },
            {
                "ticker": "D",
                "date": "2013-12-01",
                "group_index": 3,
                "label": 1,
                "label_eval_end_date": "2013-12-20",
            },
            {
                "ticker": "E",
                "date": "2013-12-20",
                "group_index": 4,
                "label": 1,
                "label_eval_end_date": "2014-01-10",
            },
            {
                "ticker": "0056",
                "date": "2014-01-02",
                "group_index": 5,
                "label": 0,
                "label_eval_end_date": "2014-02-15",
            },
            {
                "ticker": "G",
                "date": "2014-06-01",
                "group_index": 6,
                "label": 1,
                "label_eval_end_date": "2014-07-15",
            },
        ]
    )
    bundle = SimpleNamespace(
        group_table=group_table,
        profile=SimpleNamespace(
            training_label_scope="pass_only",
            training_sample_scope=TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS,
        ),
        target_valid=np.ones(len(group_table), dtype=bool),
        event_group_index=np.arange(len(group_table), dtype=np.int64),
    )
    fold = {
        "fold_id": "fold_000",
        "score_start": pd.Timestamp("2014-01-01"),
        "score_end": pd.Timestamp("2014-12-31"),
    }
    ids = _fold_group_ids(bundle, fold, validation_months=24)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_split_requires_completed_labels_before_each_information_boundary",
        ([0], [2, 3], [0, 1, 2, 3], [5, 6]),
        (
            ids["train_ids"].tolist(),
            ids["validation_ids"].tolist(),
            ids["final_ids"].tolist(),
            ids["score_ids"].tolist(),
        ),
    )

    daily_group_table = group_table.copy()
    daily_group_table["label"] = -1
    daily_bundle = SimpleNamespace(
        group_table=daily_group_table,
        profile=SimpleNamespace(
            training_label_scope=TRAINING_LABEL_SCOPE_ALL,
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        ),
        target_valid=np.ones(len(daily_group_table), dtype=bool),
        event_group_index=np.arange(len(daily_group_table), dtype=np.int64),
    )
    daily_ids = _fold_group_ids(daily_bundle, fold, validation_months=24)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_point_in_time_split_uses_all_target_valid_stock_days_and_same_label_end_embargo",
        ([0], [2, 3], [0, 1, 2, 3], [5, 6]),
        (
            daily_ids["train_ids"].tolist(),
            daily_ids["validation_ids"].tolist(),
            daily_ids["final_ids"].tolist(),
            daily_ids["score_ids"].tolist(),
        ),
    )

    inference_target_valid = np.ones(len(daily_group_table), dtype=bool)
    inference_target_valid[[3, 6]] = False
    inference_daily_bundle = SimpleNamespace(
        group_table=daily_group_table,
        profile=SimpleNamespace(
            training_label_scope=TRAINING_LABEL_SCOPE_ALL,
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        ),
        target_valid=inference_target_valid,
        event_group_index=np.arange(len(daily_group_table), dtype=np.int64),
    )
    inference_daily_ids = _fold_group_ids(
        inference_daily_bundle, fold, validation_months=24
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_pit_future_target_validity_controls_training_but_not_score_presence",
        ([0], [2], [0, 1, 2], [5, 6]),
        (
            inference_daily_ids["train_ids"].tolist(),
            inference_daily_ids["validation_ids"].tolist(),
            inference_daily_ids["final_ids"].tolist(),
            inference_daily_ids["score_ids"].tolist(),
        ),
    )

    daily_score_contract = build_score_eligibility_contract(
        DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
    )
    event_score_contract = build_score_eligibility_contract(
        STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_PAIRWISE_PROFILE
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "pit_score_eligibility_contract_is_profile_driven_and_future_target_independent",
        (
            "feature_history_only",
            False,
            True,
            "canonical_breakout_event_membership",
            False,
        ),
        (
            daily_score_contract["eligibility_basis"],
            daily_score_contract["future_target_required_for_score"],
            daily_score_contract["target_valid_required_for_training"],
            event_score_contract["eligibility_basis"],
            event_score_contract["future_target_required_for_score"],
        ),
    )

    training_contract = {
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        "continuous_target_id": "daily_opportunity_no_time_r_v1",
        "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
        "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        "seed": 42,
        "planned_periods": {
            "validation_start": "2012-01-01",
            "validation_end": "2013-12-31",
            "score_start": "2014-01-01",
            "score_end": "2014-12-31",
        },
        "observed_periods": {
            "inner_train": {"start": "2010-01-01", "end": "2011-12-31"},
            "validation": {"start": "2012-01-01", "end": "2013-11-01"},
            "final_refit": {"start": "2010-01-01", "end": "2013-11-01"},
            "score": {"start": "2014-01-01", "end": "2014-12-31"},
        },
        "model_information_cutoff": "2013-12-20",
        "group_counts": {
            "inner_train": 10,
            "validation": 20,
            "final_refit": 30,
            "score": 40,
        },
        "event_row_counts": {
            "inner_train": 10,
            "validation": 20,
            "final_refit": 30,
            "score": 40,
        },
        "model_spec": {"architecture": "inception_time_v1"},
        "experiment_settings": {"training_sample_scope": "daily_eligible_stock_days"},
        "training_settings": {"epochs_max": 200},
        "source_contract": {"target_contract": {"target_id": "daily_opportunity_no_time_r_v1"}},
        "lookahead_contract": {"score_period_used_for_training_or_epoch_selection": False},
    }
    expanded_score_contract = json.loads(json.dumps(training_contract))
    expanded_score_contract["group_counts"]["score"] = 55
    expanded_score_contract["event_row_counts"]["score"] = 55
    changed_training_contract = json.loads(json.dumps(expanded_score_contract))
    changed_training_contract["group_counts"]["final_refit"] = 31
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_pit_checkpoint_reuse_allows_score_only_expansion_but_rejects_training_change",
        (True, False),
        (
            _fold_training_contract_is_compatible(
                training_contract, expected_contract=expanded_score_contract
            ),
            _fold_training_contract_is_compatible(
                training_contract, expected_contract=changed_training_contract
            ),
        ),
    )

    class _SyntheticRankerModel:
        def load_state_dict(self, _state, strict=True):
            return self

        def to(self, _device):
            return self

        def eval(self):
            return self

    old_checkpoint_contract = json.loads(json.dumps(training_contract))
    old_checkpoint_contract.update({"schema_version": 2, "fold_id": "fold_20140101_20141231"})
    old_checkpoint_contract["group_counts"]["score"] = 1
    old_checkpoint_contract["event_row_counts"]["score"] = 1
    expected_checkpoint_contract = json.loads(json.dumps(old_checkpoint_contract))
    expected_checkpoint_contract["group_counts"]["score"] = 2
    expected_checkpoint_contract["event_row_counts"]["score"] = 2
    expected_checkpoint_contract["score_eligibility_contract"] = daily_score_contract
    synthetic_bundle = SimpleNamespace(
        profile=SimpleNamespace(
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        ),
        feature_bank=SimpleNamespace(shape=(2, 300, 10)),
        group_context=np.zeros((2, 0), dtype=np.float32),
        group_table=pd.DataFrame(
            [
                {"ticker": "2330", "date": "2014-01-02", "group_index": 0},
                {"ticker": "2317", "date": "2014-01-03", "group_index": 1},
            ]
        ),
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        rescore_fold_dir = Path(tmp_dir) / "fold_20140101_20141231"
        rescore_fold_dir.mkdir(parents=True, exist_ok=True)
        rescore_model_path = rescore_fold_dir / DEFAULT_MODEL_FILENAME
        torch.save(
            {
                "model_state_dict": {"synthetic_weight": torch.tensor([1.0])},
                "feature_count": 10,
                "context_count": 0,
                "sequence_length": 300,
                "model_spec": old_checkpoint_contract["model_spec"],
                "selected_epoch": 1,
                "fold_contract": old_checkpoint_contract,
            },
            rescore_model_path,
        )
        old_rescore_manifest = {
            **old_checkpoint_contract,
            "selected_epoch": 1,
            "epoch_selection": {"best_epoch": 1},
            "final_refit_history": [{"epoch": 1}],
            "artifacts": {
                "checkpoint": build_file_manifest(rescore_model_path),
                "scores": None,
            },
        }
        (rescore_fold_dir / "manifest.json").write_text(
            json.dumps(old_rescore_manifest), encoding="utf-8"
        )
        with patch(
            "tools.filters.breakout_quality.build_point_in_time_scores.build_model",
            return_value=_SyntheticRankerModel(),
        ), patch(
            "tools.filters.breakout_quality.build_point_in_time_scores.predict_scores",
            return_value=np.asarray([0.25, 0.75], dtype=np.float32),
        ):
            rescored = _rescore_daily_fold_from_compatible_checkpoint(
                fold_dir=rescore_fold_dir,
                bundle=synthetic_bundle,
                ids={"score_ids": np.asarray([0, 1], dtype=np.int64)},
                fold_contract=expected_checkpoint_contract,
                expected_fingerprint="expanded-score-fingerprint",
                args=SimpleNamespace(evaluation_batch_size=32),
                torch_module=torch,
                plan=SimpleNamespace(device="cpu"),
            )
        rescored_frame, rescored_manifest = rescored or (pd.DataFrame(), {})
        rewritten_checkpoint = (
            torch.load(rescore_model_path, map_location="cpu", weights_only=True)
            if rescored is not None
            else {}
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "daily_pit_checkpoint_reuse_round_trip_rescores_expanded_universe_without_refit",
        (True, 2, "expanded_daily_score_universe_checkpoint_reuse", 2, 30),
        (
            rescored is not None,
            len(rescored_frame),
            dict(rescored_manifest.get("migration") or {}).get("kind"),
            dict(rewritten_checkpoint.get("fold_contract") or {})
            .get("group_counts", {})
            .get("score"),
            dict(rewritten_checkpoint.get("fold_contract") or {})
            .get("group_counts", {})
            .get("final_refit"),
        ),
    )

    fold_contract = {
        "fold_id": "fold_000",
        "model_information_cutoff": "2013-12-20",
        "planned_periods": {
            "score_start": "2014-01-01",
            "score_end": "2014-12-31",
        },
    }
    score_frame = pd.DataFrame(
        [
            {
                "ticker": "0056",
                "date": "2014-01-02",
                "group_index": 5,
                "breakout_quality_score": 0.2,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-20",
            },
            {
                "ticker": "G",
                "date": "2014-06-01",
                "group_index": 6,
                "breakout_quality_score": 0.8,
                "fold_id": "fold_000",
                "model_information_cutoff": "2013-12-20",
            },
        ]
    )
    validated = _validate_score_frame(score_frame, fold_contract=fold_contract)
    coverage = _combined_validation(
        validated,
        bundle,
        score_start=pd.Timestamp("2014-01-01"),
        score_end=pd.Timestamp("2014-12-31"),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_score_output_has_no_future_target_and_complete_unique_coverage",
        (True, 2, 1.0, 0, 0, 0),
        (
            not bool(
                {"label", "target_raw_r", "target_daily_percentile"}
                & set(REQUIRED_SCORE_COLUMNS)
            ),
            coverage["scored_group_count"],
            coverage["coverage_rate"],
            coverage["duplicate_group_count"],
            coverage["missing_group_count"],
            coverage["extra_group_count"],
        ),
    )

    duplicate_rejected = False
    try:
        _combined_validation(
            pd.concat([validated, validated.iloc[[0]]], ignore_index=True),
            bundle,
            score_start=pd.Timestamp("2014-01-01"),
            score_end=pd.Timestamp("2014-12-31"),
        )
    except ValueError as exc:
        duplicate_rejected = "重複group" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_combined_output_rejects_duplicate_groups",
        True,
        duplicate_rejected,
    )

    identity_mismatch_rejected = False
    mismatched = validated.copy()
    mismatched.loc[mismatched["group_index"] == 5, "ticker"] = "9999"
    try:
        _combined_validation(
            mismatched,
            bundle,
            score_start=pd.Timestamp("2014-01-01"),
            score_end=pd.Timestamp("2014-12-31"),
        )
    except ValueError as exc:
        identity_mismatch_rejected = "identity不一致" in str(exc)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_combined_output_rejects_ticker_date_identity_mismatch",
        True,
        identity_mismatch_rejected,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        orderable_path = Path(tmp_dir) / "orderable.csv"
        pd.DataFrame(
            [
                {
                    "ticker": "0056",
                    "target_date": "2014-01-02",
                    "breakout_quality_score": 0.01,
                },
                {
                    "ticker": "X",
                    "target_date": "2014-01-03",
                    "breakout_quality_score": 0.99,
                },
            ]
        ).to_csv(orderable_path, index=False, encoding="utf-8-sig")
        score_dates = validated.copy()
        score_dates["date"] = pd.to_datetime(score_dates["date"], errors="raise")
        orderable = _orderable_coverage(
            score_dates,
            requested_path=str(orderable_path),
            filter_id=settings.filter_id,
            target_id=settings.continuous_target_id,
        )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_orderable_coverage_uses_canonical_target_date",
        (True, 2, 1, 0.5),
        (
            orderable["available"],
            orderable["candidate_count"],
            orderable["scored_candidate_count"],
            orderable["coverage_rate"],
        ),
        tol=1e-12,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_orderable_coverage_ignores_existing_candidate_score_column",
        (True, "selection_point_in_time_scores"),
        (
            orderable["candidate_artifact_has_existing_breakout_quality_score"],
            orderable["coverage_score_source"],
        ),
    )

    yearly_rows = [
        {
            "year": 2014,
            "group_count": 2,
            "global_spearman": 0.3,
            "mean_daily_spearman": 0.2,
            "top_bottom_target_spread": 0.8,
        }
    ]
    report_payload = {
        "status": "RESULT_AVAILABLE_PENDING_REVIEW",
        "filter_id": "breakout_quality_v1",
        "model_architecture": "inception_time_v1",
        "experiment_profile": "strategy_aligned_no_time_pass_magnitude_mse",
        "continuous_target_id": "strategy_aligned_opportunity_no_time_r_v1",
        "score_period": {"start": "2014-01-01", "end": "2014-12-31"},
        "score_coverage": {
            "expected_group_count": 2,
            "scored_group_count": 2,
            "coverage_rate": 1.0,
        },
        "workflow": {
            "training_label_scope": "pass_only",
            "seed": 42,
            "fold_count": 1,
            "fold_months": 12,
            "inner_validation_months": 24,
        },
        "metrics": {
            "pass_only_target": {
                "group_count": 2,
                "global_spearman": 0.3,
                "mean_daily_spearman": 0.2,
                "top_decile_target_mean": 1.5,
                "bottom_decile_target_mean": 0.7,
                "top_bottom_target_spread": 0.8,
            },
            "all_valid_target": {
                "group_count": 2,
                "global_spearman": 0.25,
                "mean_daily_spearman": 0.15,
                "top_decile_target_mean": 1.4,
                "bottom_decile_target_mean": 0.6,
                "top_bottom_target_spread": 0.8,
            },
        },
        "yearly_pass_only": yearly_rows,
        "direction_summary": _direction_summary(yearly_rows),
        "fold_metrics": [
            {
                "fold_id": "fold_000",
                "group_count": 2,
                "score_mean": 0.5,
                "score_std": 0.2,
                "score_p10": 0.3,
                "score_p50": 0.5,
                "score_p90": 0.7,
                "adjacent_mean_shift_in_pooled_std": None,
                "pass_target_spearman": 0.3,
            }
        ],
        "fold_drift": {
            "criterion": "synthetic drift contract",
            "drift_flag": False,
            "flagged_folds": [],
            "max_adjacent_mean_shift_in_pooled_std": 0.0,
        },
        "classification_overlap": {
            "score_vs_pass_reject_auc": 0.6,
            "overall_pass_share": 0.5,
            "top_score_decile_pass_share": 1.0,
            "interpretation_contract": "synthetic overlap contract",
        },
        "orderable_candidate_coverage": {
            "available": True,
            "path": "/tmp/orderable.csv",
            "candidate_count": 2,
            "scored_candidate_count": 2,
            "unscored_candidate_count": 0,
            "coverage_rate": 1.0,
            "coverage_score_source": "selection_point_in_time_scores",
        },
        "source_artifacts": {
            "point_in_time_manifest": "/tmp/manifest.json",
            "point_in_time_scores": "/tmp/scores.csv",
            "point_in_time_coverage": "/tmp/coverage.csv",
        },
        "report_artifacts": {
            "markdown": "/tmp/audit.md",
            "json": "/tmp/audit.json",
        },
    }
    report_console = render_point_in_time_console(report_payload)
    colored_report_console = render_point_in_time_console(report_payload, color=True)
    compact_report_console = render_point_in_time_compact_console(report_payload)
    report_markdown = render_point_in_time_markdown(report_payload)
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_outputs_readable_console_summary",
        True,
        all(
            text in report_console
            for text in (
                "Selection Point-in-time 模型評估報表",
                "核心排序能力",
                "年度穩定性",
                "Fold 分布與漂移",
                "策略 optimizer：未執行",
            )
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_console_uses_shared_status_colors",
        True,
        all(
            token in colored_report_console
            for token in (
                "\x1b[96m",
                "\x1b[92m",
                "\x1b[93m",
            )
        )
        and "\x1b[" not in report_console,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_compact_console_avoids_repeated_detail_tables",
        True,
        all(
            text in compact_report_console
            for text in (
                "PIT 模型驗證",
                "核心排序",
                "年度穩定",
                "模型 Gate",
                "執行策略績效驗證",
            )
        )
        and all(
            text not in compact_report_console
            for text in (
                "Current Breakout Quality Workflow",
                "Fold 分布與漂移",
                "fold_000",
                "完整指標 JSON",
            )
        ),
    )
    project_root = Path(__file__).resolve().parents[2]
    ranker_source = (
        project_root / "services" / "breakout_quality" / "train_continuous_ranker.py"
    ).read_text(encoding="utf-8")
    pit_builder_source = (
        project_root / "services" / "breakout_quality" / "point_in_time_scores.py"
    ).read_text(encoding="utf-8")
    pit_builder_tree = ast.parse(pit_builder_source)
    pit_builder_call_keywords = {
        node.func.id: {keyword.arg for keyword in node.keywords if keyword.arg}
        for node in ast.walk(pit_builder_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"_load_reusable_fold", "_migrate_compatible_legacy_fold"}
    }
    base_target_audit_source = (
        project_root / "tools" / "audit" / "breakout_quality" / "continuous_target.py"
    ).read_text(encoding="utf-8")
    no_time_target_source = (
        project_root / "tools" / "audit" / "breakout_quality" / "no_time_continuous_target.py"
    ).read_text(encoding="utf-8")
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_resume_wires_torch_only_to_legacy_migration",
        (False, True),
        (
            "torch_module" in pit_builder_call_keywords.get("_load_reusable_fold", set()),
            "torch_module" in pit_builder_call_keywords.get(
                "_migrate_compatible_legacy_fold", set()
            ),
        ),
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_compact_training_output_is_one_line_per_new_fold",
        True,
        ranker_source.count("if not compact_console:") >= 4
        and "fold_progress.print_line(" in pit_builder_source
        and "best epoch=" in pit_builder_source
        and "PIT Scores 完成" in pit_builder_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_compact_target_output_hides_base_audit_and_summarizes_final_target",
        True,
        "if compact_console_enabled():" in base_target_audit_source
        and "Continuous Target 完成" in no_time_target_source
        and "Selection mean=" in no_time_target_source
        and "OOS未評估" in no_time_target_source,
    )
    add_check(
        results,
        "synthetic_breakout_quality",
        case_id,
        "point_in_time_audit_outputs_complete_markdown_report",
        True,
        all(
            text in report_markdown
            for text in (
                "# Breakout Quality Selection Point-in-time 模型評估報表",
                "## 2. 核心排序能力",
                "## 3. 年度穩定性（PASS-only）",
                "## 4. Fold 分布與漂移",
                "## 7. 研究邊界與下一步",
                "## 8. 工件",
            )
        ),
    )

    summary["workflow"] = "selection_point_in_time_scores"
    summary["score_contract"] = "future_target_excluded"
    return results, summary

def validate_breakout_quality_selection_point_in_time_score_sort_contract_case(_base_params):
    case_id = "BREAKOUT_QUALITY_SELECTION_POINT_IN_TIME_SCORE_SORT"
    results = []
    summary = {"ticker": case_id, "synthetic": True}

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
    from filters.breakout_quality.strategy_compare_engine import (
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_score_sort_desc_tie_and_missing_fallback_contract",
        ["A", "C", "D", "B"], [row["ticker"] for row in ranking_rows],
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_audit_is_bound_to_exact_source_artifact_hashes",
        (True, True), (exact_binding_accepted, stale_binding_rejected),
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
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
    add_check(
        results, "synthetic_breakout_quality", case_id,
        "point_in_time_strategy_diagnostic_rejects_true_score_mismatch_with_identity",
        (True, True),
        (true_score_mismatch_rejected, mismatch_message_has_identity),
    )

    summary["workflow"] = "selection_point_in_time_scores"
    summary["score_contract"] = "future_target_excluded"
    summary["strategy_score_sort"] = "missing_fallback_original_buy_sort"
    return results, summary
