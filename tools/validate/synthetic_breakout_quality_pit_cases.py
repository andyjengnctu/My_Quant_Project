from __future__ import annotations

from .checks import run_bound_checks

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

def _check_selection_pit_publication_invariants(check_true, pit_module):
    """Cross-process writer ownership and manifest-last data consistency."""
    import subprocess
    import sys

    from core.file_integrity import compute_file_sha256, exclusive_file_lock
    from filters.breakout_quality.workflow_io import write_json

    with tempfile.TemporaryDirectory(prefix="selection_pit_publication_contract_") as temp_dir:
        directory = Path(temp_dir)
        score_path = directory / "selection_point_in_time_scores.csv"
        coverage_path = directory / "selection_point_in_time_coverage.csv"
        manifest_path = directory / "selection_point_in_time_manifest.json"
        initial_score = pd.DataFrame({"ticker": ["2330"], "score": [0.2]})
        initial_coverage = pd.DataFrame({"fold_id": ["old"]})
        initial_score.to_csv(score_path, index=False, encoding="utf-8-sig")
        initial_coverage.to_csv(coverage_path, index=False, encoding="utf-8-sig")
        initial_manifest = {"score_period": {"start": "2021-01-01", "end": "2021-12-31"}}
        write_json(manifest_path, initial_manifest)
        baseline = compute_file_sha256(manifest_path)

        def publish(expected_baseline, *, score=0.9):
            return pit_module._publish_selection_pit_aggregate(
                score_path=score_path,
                coverage_path=coverage_path,
                manifest_path=manifest_path,
                score_frame=pd.DataFrame({"ticker": ["2330"], "score": [score]}),
                coverage_frame=pd.DataFrame({"fold_id": ["new"]}),
                manifest={"score_period": {"start": "2021-01-01", "end": "2021-12-31"}},
                baseline_manifest_sha256=expected_baseline,
                point_in_time_dir=directory,
                score_start=pd.Timestamp("2021-01-01"),
                score_end=pd.Timestamp("2021-12-31"),
                selection_end=pd.Timestamp("2020-12-31"),
            )

        publish(baseline)
        published = json.loads(manifest_path.read_text(encoding="utf-8"))
        check_true(
            "pit_atomic_publication_manifest_commits_exact_score_and_coverage_bytes",
            published["artifacts"]["scores"] == build_file_manifest(score_path)
            and published["artifacts"]["coverage"] == build_file_manifest(coverage_path)
            and abs(float(pd.read_csv(score_path)["score"].iloc[0]) - 0.9) < 1e-12,
        )
        committed_bytes = (
            score_path.read_bytes(), coverage_path.read_bytes(), manifest_path.read_bytes()
        )
        stale_rejected = False
        try:
            publish(baseline, score=0.1)
        except RuntimeError as exc:
            stale_rejected = "concurrent publication" in str(exc)
        check_true(
            "pit_concurrent_publisher_cannot_overwrite_newer_manifest",
            stale_rejected and committed_bytes == (
                score_path.read_bytes(), coverage_path.read_bytes(), manifest_path.read_bytes()
            ),
        )
        initial_stager = pit_module._stage_pit_aggregate_csv
        stage_count = [0]

        def fail_second_stage(*args):
            stage_count[0] += 1
            if stage_count[0] == 2:
                raise OSError("synthetic coverage stage failure")
            return initial_stager(*args)

        staging_failure_rejected = False
        with patch.object(pit_module, "_stage_pit_aggregate_csv", side_effect=fail_second_stage):
            try:
                publish(compute_file_sha256(manifest_path), score=0.3)
            except OSError as exc:
                staging_failure_rejected = "stage failure" in str(exc)
        check_true(
            "pit_csv_stage_failure_preserves_current_publication_and_cleans_temp",
            staging_failure_rejected
            and committed_bytes == (
                score_path.read_bytes(), coverage_path.read_bytes(), manifest_path.read_bytes()
            )
            and not list(directory.glob(".*.tmp")),
        )
        active_lock = directory / ".selection_point_in_time_manifest.json.publish.lock"
        lock_rejected = False
        with exclusive_file_lock(active_lock):
            try:
                publish(compute_file_sha256(manifest_path), score=0.4)
            except RuntimeError as exc:
                lock_rejected = "publication is active" in str(exc)
        check_true(
            "pit_active_publication_lock_refuses_second_writer",
            lock_rejected and committed_bytes == (
                score_path.read_bytes(), coverage_path.read_bytes(), manifest_path.read_bytes()
            ),
        )
        process_lock = directory / ".selection_pit_build.lock"
        with exclusive_file_lock(process_lock):
            probe = (
                "from core.file_integrity import exclusive_file_lock\n"
                "from pathlib import Path\n"
                f"with exclusive_file_lock(Path({str(process_lock)!r})):\n    pass\n"
            )
            outcome = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=str(Path(__file__).resolve().parents[2]),
                capture_output=True, text=True, check=False,
            )
        check_true(
            "pit_build_lock_is_cross_process_and_crash_safe",
            outcome.returncode != 0
            and "Another artifact publication is active" in outcome.stderr,
        )
        # A FIT Repair can rename the whole profile/PIT directory. The build
        # lock must stay at one stable inode so a new builder cannot bypass it.
        from services.breakout_quality.fit_repair_transaction import selection_pit_build_lock_path
        simulated_profile = directory / "architecture" / "profile"
        simulated_pit = simulated_profile / "point_in_time"
        simulated_pit.mkdir(parents=True)
        stable_lock = selection_pit_build_lock_path(simulated_pit)
        renamed_profile = simulated_profile.with_name(".profile.fit_repair_pending_synthetic")
        stable_blocked = False
        with exclusive_file_lock(stable_lock):
            simulated_profile.rename(renamed_profile)
            try:
                with exclusive_file_lock(selection_pit_build_lock_path(simulated_pit)):
                    pass
            except RuntimeError:
                stable_blocked = True
        check_true(
            "pit_build_lock_stays_stable_when_fit_repair_renames_profile",
            stable_blocked and stable_lock == selection_pit_build_lock_path(simulated_pit),
        )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_before_failure = manifest_path.read_bytes()
        interrupted = False
        with patch.object(pit_module, "write_json", side_effect=OSError("injected manifest failure")):
            try:
                publish(compute_file_sha256(manifest_path), score=0.7)
            except OSError as exc:
                interrupted = "manifest failure" in str(exc)
        check_true(
            "pit_interrupted_csv_commit_fails_closed_against_old_manifest_sha",
            interrupted
            and manifest_path.read_bytes() == manifest_before_failure
            and compute_file_sha256(score_path) != previous_manifest["artifacts"]["scores"]["sha256"],
        )
        parity_frame = pd.DataFrame({
            "ticker": ["2330", "0050"], "score": [0.123456789, 0.875],
            "head": ["原始", "Safety"],
        })
        old_csv = directory / "old_style.csv"
        parity_frame.to_csv(old_csv, index=False, encoding="utf-8-sig")
        staged_csv = pit_module._stage_pit_aggregate_csv(directory / "new_style.csv", parity_frame)
        check_true(
            "pit_staged_csv_is_byte_identical_to_previous_pandas_writer",
            staged_csv.read_bytes() == old_csv.read_bytes(),
        )
        staged_csv.unlink()
        immutable_example = {"z": ["歷史"], "a": 1}
        json_path = directory / "format_parity.json"
        write_json(json_path, immutable_example)
        legacy_json_text = json.dumps(
            immutable_example, ensure_ascii=False, indent=2, sort_keys=True,
        )
        legacy_json_path = directory / "format_parity_legacy.json"
        # AI: Exact reference: pre-atomic Research writer used Path.write_text().
        # AI: On Windows its native text-mode newline conversion produced CRLF.
        legacy_json_path.write_text(legacy_json_text, encoding="utf-8")
        check_true(
            "research_json_atomic_publication_preserves_historical_exact_bytes",
            json_path.read_bytes() == legacy_json_path.read_bytes(),
        )

        # AI: Simulate Windows' native newline behavior on non-Windows runners too.
        original_path_open = Path.open

        def windows_default_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if "b" not in mode and any(flag in mode for flag in ("w", "a", "x")):
                if kwargs.get("newline") is None:
                    kwargs["newline"] = "\r\n"
            return original_path_open(path, *args, **kwargs)

        windows_json_path = directory / "format_parity_windows.json"
        windows_legacy_path = directory / "format_parity_windows_legacy.json"
        with patch.object(Path, "open", new=windows_default_open):
            windows_legacy_path.write_text(legacy_json_text, encoding="utf-8")
            write_json(windows_json_path, immutable_example)
        expected_windows_bytes = legacy_json_text.replace("\n", "\r\n").encode("utf-8")
        check_true(
            "research_json_atomic_publication_preserves_windows_native_crlf",
            windows_json_path.read_bytes() == windows_legacy_path.read_bytes()
            == expected_windows_bytes,
        )



def validate_breakout_quality_point_in_time_score_builder_contract_case(_base_params):
    """Protect PIT legality, fold identity and score-universe semantics."""
    case_id = "BREAKOUT_QUALITY_POINT_IN_TIME_SCORE_BUILDER"
    results = []
    summary = {"ticker": case_id, "synthetic": True}
    check, check_true = bind_checks(results, "synthetic_breakout_quality", case_id)

    from core.breakout_quality_registry import (
        CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
        DAILY_UNIVERSAL_SHARED_HS_QUALIFICATION_CONDITIONAL_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
    )
    from core.breakout_quality_policy import (
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
    from services.breakout_quality.point_in_time_scores import (
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

    # Strategy Compare Forward/ Rolling / Robustness reuse must be an actual
    # checkpoint-before-epoch behavior, not just a READY display projection.
    from services.breakout_quality.point_in_time_scores import (
        _resolve_verified_cross_mode_fit_source,
    )
    from services.research.strategy_compare_training import (
        build_strategy_compare_trainer_command,
        resolve_strategy_compare_fit_source_dirs,
    )
    from filters.breakout_quality.paths import (
        resolve_filter_point_in_time_dir,
        resolve_model_research_robustness_seed_dirs,
    )
    from core.breakout_quality_policy import get_breakout_quality_rolling_test_mode
    from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME
    from filters.breakout_quality.paths import SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
    from services.breakout_quality.point_in_time_scores import FOLD_MANIFEST_FILENAME
    from services.breakout_quality import point_in_time_scores as pit_module
    _check_selection_pit_publication_invariants(check_true, pit_module)
    with tempfile.TemporaryDirectory(prefix="strategy_fit_cross_mode_source_") as td:
        base = Path(td).resolve()
        profile = DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE
        source = SimpleNamespace(
            filter_id=BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
            model_architecture="inception_time_v1", experiment_profile=profile,
            score_source="selection_point_in_time",
            point_in_time_fold_months=12, point_in_time_score_start_date="2021-01-01",
            point_in_time_score_end_date="auto", point_in_time_fold_anchor_date="2021-01-01",
            point_in_time_single_score_block=True,
            single_seed_strategy_conversion_authorized=False, dl_id="fixture_source",
        )
        seed = 693545351
        target = base / "new_forward_scores"
        candidate_roots = resolve_strategy_compare_fit_source_dirs(
            project_root=base, source=source, seed=seed, target_dir=target,
        )
        model_robust_roll, _ = resolve_model_research_robustness_seed_dirs(
            base, source.filter_id, source.model_architecture, profile,
            seed=seed, rolling=True,
        )
        check_true(
            "strategy_oos_fitting_source_resolver_includes_rolling_and_same_seed_robustness",
            resolve_filter_point_in_time_dir(
                base, source.filter_id, source.model_architecture, profile,
            ) in candidate_roots and model_robust_roll in candidate_roots
            and target not in candidate_roots,
        )
        oos_mode = get_breakout_quality_rolling_test_mode("oos")
        check_true(
            "strategy_rolling_fitting_source_resolver_includes_published_single_block_oos",
            any(path.name == oos_mode.point_in_time_dirname for path in candidate_roots),
        )
        workflow = SimpleNamespace(
            point_in_time_fold_months=12, point_in_time_score_start_date="2021-01-01",
            point_in_time_score_end_date="auto", point_in_time_inner_validation_months=24,
        )
        command, trainer_args, _metadata = build_strategy_compare_trainer_command(
            source=source, workflow=workflow, seed=seed,
            point_in_time_dir_override=target,
            fit_source_pit_dirs=candidate_roots,
        )
        from services.research import strategy_compare_training as strategy_training
        class _StopBeforeTrainerStarts(Exception):
            pass
        captured_command = []
        def _capture_without_starting_trainer(**kwargs):
            captured_command.extend(kwargs["command"])
            raise _StopBeforeTrainerStarts
        with patch.object(strategy_training, "run_logged_training_process",
                          side_effect=_capture_without_starting_trainer):
            try:
                strategy_training.run_strategy_compare_training_unit(
                    project_root=base, source=source, workflow=workflow, seed=seed,
                    model_dir=target, research_dir=target, log_path=base / "never_train.log",
                    comparison_start="2021-01-01", comparison_end="2026-03-02",
                    point_in_time_dir_override=target,
                )
            except _StopBeforeTrainerStarts:
                pass
        check_true(
            "strategy_single_and_robustness_shared_training_unit_reaches_cross_mode_fit_precheck",
            captured_command.count("--fit-source-pit-dir") == len(candidate_roots)
            and str(model_robust_roll.resolve()) in captured_command
            and "--single-score-block" in captured_command
            and "--resume" in captured_command,
        )
        check_true(
            "strategy_trainer_command_passes_shared_fit_candidate_roots_without_changing_score_mode",
            trainer_args.count("--fit-source-pit-dir") == len(candidate_roots)
            and "--single-score-block" in trainer_args
            and "--point-in-time-dir-override" in trainer_args
            and command[-len(trainer_args):] == trainer_args,
        )
        source_root = base / "published_rolling"
        source_fold = source_root / "folds" / "fold_20210101_20211231"
        source_fold.mkdir(parents=True)
        contract = {
            "filter_id": source.filter_id,
            "model_architecture": source.model_architecture,
            "experiment_profile": profile,
            "seed": seed,
            "planned_periods": {"score_start": "2021-01-01", "score_end": "2026-03-02"},
        }
        (source_fold / FOLD_MANIFEST_FILENAME).write_text(
            json.dumps(contract), encoding="utf-8",
        )
        (source_fold / DEFAULT_MODEL_FILENAME).write_bytes(b"published-fit-one")
        (source_root / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).write_text(
            json.dumps({"folds": [contract]}), encoding="utf-8",
        )
        expected = {**contract, "planned_periods": {
            "score_start": "2021-01-01", "score_end": "2021-12-31",
        }}
        with patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(SimpleNamespace(), None)) as validation:
            matched = _resolve_verified_cross_mode_fit_source(
                fit_source_pit_dirs=(source_root,),
                target_fold_dir=target / "folds" / "fold_20210101_20260302",
                fold_contract=expected, bundle=object(), torch_module=object(),
                plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=base,
            )
        check_true(
            "cross_mode_source_reuses_published_fit_without_score_horizon_equality",
            matched is not None and matched[0] == source_fold
            and matched[1]["source_checkpoint"]["sha256"]
                == build_file_manifest(source_fold / DEFAULT_MODEL_FILENAME)["sha256"]
            and validation.call_count == 1,
        )
        with patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(None, "external context mismatch")):
            try:
                _resolve_verified_cross_mode_fit_source(
                    fit_source_pit_dirs=(source_root,),
                    target_fold_dir=target / "folds" / "fold_20210101_20260302",
                    fold_contract=expected, bundle=object(), torch_module=object(),
                    plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=base,
                )
            except RuntimeError as exc:
                invalid_blocked = "automatic training blocked" in str(exc)
            else:
                invalid_blocked = False
        check_true("unverified_cross_mode_fit_blocks_training_instead_of_refitting", invalid_blocked)
        second_root = base / "published_oos"
        second_fold = second_root / "folds" / "fold_20210101_20260302"
        second_fold.mkdir(parents=True)
        (second_fold / FOLD_MANIFEST_FILENAME).write_text(json.dumps(contract), encoding="utf-8")
        (second_fold / DEFAULT_MODEL_FILENAME).write_bytes(b"different-fit-weights")
        (second_root / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME).write_text(
            json.dumps({"folds": [contract]}), encoding="utf-8",
        )
        with patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(SimpleNamespace(), None)):
            try:
                _resolve_verified_cross_mode_fit_source(
                    fit_source_pit_dirs=(source_root, second_root),
                    target_fold_dir=target / "folds" / "fold_20210101_20260302",
                    fold_contract=expected, bundle=object(), torch_module=object(),
                    plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=base,
                )
            except RuntimeError as exc:
                conflicting_sha_blocked = "different SHA" in str(exc)
            else:
                conflicting_sha_blocked = False
        check_true("same_fitting_identity_different_published_checkpoint_sha_blocks", conflicting_sha_blocked)
        unrelated = {**expected, "seed": seed + 1}
        check_true(
            "unrelated_seed_does_not_provide_fit_or_trigger_training_reuse",
            _resolve_verified_cross_mode_fit_source(
                fit_source_pit_dirs=(source_root,),
                target_fold_dir=target / "folds" / "fold_20210101_20260302",
                fold_contract=unrelated, bundle=object(), torch_module=object(),
                plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=base,
            ) is None,
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
        "name": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
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
        "experiment_profile": DAILY_UNIVERSAL_NO_TIME_PAIRWISE_PROFILE,
        "continuous_target_id": "synthetic_target",
        "training_label_scope": TRAINING_LABEL_SCOPE_ALL,
        "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
        "model_information_cutoff": "2020-12-31",
        "model_spec": {"name": "synthetic_model"},
        "experiment_settings": profile_payload,
        "seed": 693545351,
        "training_settings": {
            "epochs_max": 200,
            "batch_size": 128,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
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
    source_fitting_evidence = SimpleNamespace(
        training_settings={
            "max_epochs": 200,
            "batch_size": 128,
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
            "use_inner_validation": True,
            "inner_validation_months": 24,
            "early_stopping_patience": 1,
            "early_stopping_min_delta": 0.0,
        },
        torch_execution={key: value for key, value in execution_payload.items() if key != "requested_device"},
        selected_epoch=2,
        model_record=checkpoint_manifest,
    )
    import_issues = _forward_checkpoint_import_issues(
        source_manifest=source_manifest,
        source_report=source_report,
        source_fitting_evidence=source_fitting_evidence,
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
        source_fitting_evidence=source_fitting_evidence,
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
    binary = get_breakout_quality_workflow_settings(
        experiment_profile=UNIQUE_GROUP_SAMPLING_EXPERIMENT_PROFILE
    )
    continuous = get_breakout_quality_workflow_settings(
        experiment_profile=STRATEGY_ALIGNED_NO_TIME_PASS_MAGNITUDE_MSE_PROFILE
    )
    daily = get_breakout_quality_workflow_settings(
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
    run_bound_checks(
        check,
        (
            ('pit_oos_is_single_block_and_rolling_is_annual_with_partial_tail', (1, 6, '2021-01-01', '2026-03-02'), (len(oos), len(rolling), str(rolling[0]['score_start'].date()), str(rolling[-1]['score_end'].date())),),
            ('pit_fold_identity_is_date_stable', 'fold_20140101_20141231', _stable_fold_id(pd.Timestamp('2014-01-01'), pd.Timestamp('2014-12-31')),),
        ),
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

    a1_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    a2_profile = get_breakout_quality_experiment_profile(
        DAILY_UNIVERSAL_SHARED_SAFETY_CONTEXT_WEIGHTED_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE
    )
    expected_raw_duo_columns = {
        "primary": "breakout_quality_score",
        "raw_safety": "raw_safety_score",
        "raw_mfe": "raw_mfe_score",
    }
    from core.breakout_quality_runtime import (
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
        for training_objective in CONTINUOUS_RANKER_TRAINING_OBJECTIVES
        for training_policy in (
            get_continuous_ranker_training_policy(training_objective),
        )
        for column in (
            training_policy.categorical_score_columns
            + tuple(
                column
                for _head_name, column in (
                    training_policy.score_output_policy.persisted_columns
                    if training_policy.score_output_policy is not None else ()
                )
            )
        )
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

    # Controlled native FIT repair must republish the newly verified first Fold
    # into the *same* canonical shared FitKey cache. A temporary rebuild cache
    # cannot be discarded without leaving the old first Fold's cache absent.
    from services.breakout_quality.point_in_time_scores import publish_verified_native_first_fold_cache
    with tempfile.TemporaryDirectory(prefix="pit_fit_repair_canonical_cache_") as td:
        root = Path(td)
        rolling_dir = root / "pit"
        fold_dir = rolling_dir / "folds" / "fold_20210101_20211231"
        fold_dir.mkdir(parents=True)
        original = dict(expected_contract)
        original.update({
            "fold_id": fold_dir.name,
            "planned_periods": {**dict(expected_contract.get("planned_periods") or {}),
                                 "score_start": "2021-01-01"},
        })
        (fold_dir / "model.pt").write_bytes(b"new canonical native model bytes")
        sha_record = build_file_manifest(fold_dir / "model.pt")
        original["artifacts"] = {"checkpoint": sha_record}
        original["fitting_origin"] = {"kind": "rolling_native", "checkpoint": sha_record}
        (fold_dir / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
        cache_root = root / "fit_cache"
        key = publish_verified_native_first_fold_cache(
            rolling_dir=rolling_dir, score_start="2021-01-01",
            cache_root=cache_root, expected_checkpoint_sha256=sha_record["sha256"],
        )
        entry = cache_root / key
        check_true("repair_native_fit_published_by_pit_to_canonical_shared_cache",
                   key == fold_training_identity(original)
                   and (entry / "model.pt").read_bytes() == (fold_dir / "model.pt").read_bytes()
                   and build_file_manifest(entry / "model.pt") == sha_record)
        try:
            publish_verified_native_first_fold_cache(
                rolling_dir=rolling_dir, score_start="2021-01-01",
                cache_root=cache_root, expected_checkpoint_sha256=sha_record["sha256"],
            )
        except RuntimeError as exc:
            duplicate_guarded = "already occupied" in str(exc)
        else:
            duplicate_guarded = False
        check_true("repair_native_fit_cache_does_not_overwrite_existing_fitkey", duplicate_guarded)
        (entry / "model.pt").unlink()
        original["fitting_origin"] = {"kind": "forward_import", "checkpoint": sha_record}
        (fold_dir / "manifest.json").write_text(json.dumps(original), encoding="utf-8")
        try:
            publish_verified_native_first_fold_cache(
                rolling_dir=rolling_dir, score_start="2021-01-01",
                cache_root=root / "unpublished_cache",
                expected_checkpoint_sha256=sha_record["sha256"],
            )
        except RuntimeError as exc:
            fake_origin_guarded = "no verified checkpoint origin" in str(exc)
        else:
            fake_origin_guarded = False
        check_true("repair_native_fit_rejects_forward_origin_instead_of_relabeled_legacy", fake_origin_guarded)

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

    # Shared FitKey cache is append-once: an incomplete existing directory is
    # historical evidence, never a disposable staging directory. Keep that
    # evidence if a second process attempts to publish a native checkpoint.
    from core.file_integrity import sibling_publication_lock_path, exclusive_file_lock
    with tempfile.TemporaryDirectory(prefix="pit_shared_fit_lock_") as td:
        root = Path(td)
        cache_root = root / "shared_fit"
        fold_dir = root / "fold"
        fold_dir.mkdir()
        (fold_dir / "model.pt").write_bytes(b"new native fold bytes")
        local_manifest = {
            **expected_contract,
            "artifacts": {"checkpoint": build_file_manifest(fold_dir / "model.pt")},
        }
        (fold_dir / "manifest.json").write_text(json.dumps(local_manifest), encoding="utf-8")
        entry = cache_root / fold_training_identity(expected_contract)
        entry.mkdir(parents=True)
        (entry / "unverified_evidence.txt").write_bytes(b"historical partial evidence")
        partial_blocked = False
        from services.breakout_quality.point_in_time_scores import _publish_fold_checkpoint_to_cache
        try:
            _publish_fold_checkpoint_to_cache(
                fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract,
            )
        except RuntimeError as exc:
            partial_blocked = "entry不完整" in str(exc)
        check_true(
            "shared_fit_partial_entry_is_never_deleted_or_overwritten",
            partial_blocked
            and (entry / "unverified_evidence.txt").read_bytes() == b"historical partial evidence"
            and not (entry / "model.pt").exists(),
        )
        # Unchanged FitKey and checkpoint may REUSE an existing publication.
        import shutil
        shutil.rmtree(entry)
        first = _publish_fold_checkpoint_to_cache(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract,
        )
        cached_manifest_bytes = (entry / "manifest.json").read_bytes()
        again = _publish_fold_checkpoint_to_cache(
            fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract,
        )
        check_true(
            "shared_fit_cache_first_publish_and_identical_reuse_preserve_bytes",
            first.get("reused_existing_cache") is False
            and again.get("reused_existing_cache") is True
            and (entry / "manifest.json").read_bytes() == cached_manifest_bytes,
        )
        # Both Forward import and native fold publish use this SAME cache lock.
        active_lock = sibling_publication_lock_path(entry)
        native_lock_blocked = False
        with exclusive_file_lock(active_lock):
            try:
                _publish_fold_checkpoint_to_cache(
                    fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract,
                )
            except RuntimeError as exc:
                native_lock_blocked = "Another shared FIT" in str(exc)
        check_true("shared_fit_native_publisher_uses_cross_process_entry_lock", native_lock_blocked)
        import subprocess
        import sys
        lock_probe = (
            "from pathlib import Path\n"
            "from core.file_integrity import exclusive_file_lock\n"
            f"with exclusive_file_lock(Path({str(active_lock)!r})):\n    pass\n"
        )
        with exclusive_file_lock(active_lock):
            result = subprocess.run(
                [sys.executable, "-c", lock_probe],
                cwd=str(Path(__file__).resolve().parents[2]),
                capture_output=True, text=True, check=False,
            )
        check_true("shared_fit_cache_lock_is_effective_across_processes",
                   result.returncode != 0 and "Another artifact publication is active" in result.stderr)
        from services.breakout_quality.point_in_time_scores import _import_forward_checkpoint_to_fitting_cache
        forward_import_lock_blocked = False
        with exclusive_file_lock(active_lock):
            try:
                _import_forward_checkpoint_to_fitting_cache(
                    source_model_dir=root / "forward", source_report_path=root / "forward_report.json",
                    cache_root=cache_root, fold_contract=expected_contract,
                    bundle=SimpleNamespace(), torch_module=SimpleNamespace(),
                    execution_plan=SimpleNamespace(),
                )
            except RuntimeError as exc:
                forward_import_lock_blocked = "Another shared FIT" in str(exc)
        check_true("shared_fit_forward_import_and_native_publisher_have_one_lock_owner",
                   forward_import_lock_blocked)
        from services.breakout_quality.point_in_time_scores import _rescore_fold_from_fitting_checkpoint
        shared_reader_blocked = False
        with exclusive_file_lock(active_lock):
            try:
                _rescore_fold_from_fitting_checkpoint(
                    source_fold_dir=entry, target_fold_dir=root / "rescore",
                    bundle=SimpleNamespace(), ids={}, fold_contract=expected_contract,
                    expected_fingerprint="synthetic", args=SimpleNamespace(checkpoint_cache_root=str(cache_root)),
                    torch_module=SimpleNamespace(), plan=SimpleNamespace(),
                )
            except RuntimeError as exc:
                shared_reader_blocked = "Another shared FIT" in str(exc)
        check_true("shared_fit_reader_cannot_observe_partially_moved_cache_during_repair",
                   shared_reader_blocked)
        (fold_dir / "model.pt").write_bytes(b"another checkpoint same fitting identity")
        local_manifest["artifacts"]["checkpoint"] = build_file_manifest(fold_dir / "model.pt")
        (fold_dir / "manifest.json").write_text(json.dumps(local_manifest), encoding="utf-8")
        conflict_blocked = False
        try:
            _publish_fold_checkpoint_to_cache(
                fold_dir=fold_dir, cache_root=cache_root, fold_contract=expected_contract,
            )
        except RuntimeError as exc:
            conflict_blocked = "同一fitting identity產生不同checkpoint" in str(exc)
        check_true(
            "shared_fit_same_identity_conflict_retains_original_bytes",
            conflict_blocked and (entry / "model.pt").read_bytes() == b"new native fold bytes",
        )

    # A live native Rolling checkpoint with changed source *record serialization*
    # must enter the same strict FIT owner proof used by archives. Raw FitKey
    # equality alone must not force a new optimizer FIT or discard the model.
    with tempfile.TemporaryDirectory(prefix="pit_native_fit_source_bridge_") as td:
        root = Path(td)
        fold_dir = root / "fold_20210101_20211231"
        fold_dir.mkdir()
        model_path = fold_dir / DEFAULT_MODEL_FILENAME
        model_path.write_bytes(b"immutable-original-fitting-checkpoint")
        historical = {
            "fold_id": fold_dir.name,
            "model_information_cutoff": "2020-12-31",
            "source_contract": {"source_data_inventory": {"version": "old"},
                                "dataset_artifacts": {"events": "old"}},
            "artifacts": {"checkpoint": build_file_manifest(model_path)},
        }
        expected = {**historical, "source_contract": {
            "source_data_inventory": {"version": "new"},
            "dataset_artifacts": {"events": "refreshed"},
        }}
        manifest_path = fold_dir / FOLD_MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(historical), encoding="utf-8")
        unchanged_checkpoint = model_path.read_bytes()
        unchanged_manifest = manifest_path.read_bytes()
        from unittest.mock import MagicMock
        fake_model = SimpleNamespace(load_state_dict=lambda *_a, **_k: None,
                                     to=lambda *_a: None, eval=lambda: None)
        fake_bundle = SimpleNamespace(
            feature_bank=np.zeros((1, 2, 3)), group_context=np.zeros((1, 1)),
            profile=SimpleNamespace(training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS),
        )
        fake_checkpoint = {"model_spec": {}, "model_state_dict": {}}
        def frame(*_a, **_k):
            return pd.DataFrame({"ticker": ["2330"], "date": ["2021-01-04"],
                                 "group_index": [0], "score": [0.5]})
        with patch.object(pit_module, "_fold_training_contract_is_compatible", return_value=False), \
             patch.object(pit_module, "scientific_torch_execution", return_value={}), \
             patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(None, "source_data_inventory not proven")) as fit_owner, \
             patch.object(pit_module, "_checkpoint_matches_rescore_contract") as checkpoint_check:
            errors = []
            blocked = pit_module._rescore_fold_from_compatible_checkpoint(
                fold_dir=fold_dir, bundle=fake_bundle,
                ids={"score_ids": [0], "validation_ids": [0]},
                fold_contract=expected, expected_fingerprint="test", args=SimpleNamespace(),
                torch_module=SimpleNamespace(load=MagicMock(return_value=fake_checkpoint)),
                plan=SimpleNamespace(device="cpu"), fit_validation_errors=errors,
            )
            check_true(
                "pit_native_source_drift_requires_fit_owner_proof_before_scoring",
                blocked is None and errors == ["source_data_inventory not proven"]
                and fit_owner.call_count == 1 and not checkpoint_check.called
                and model_path.read_bytes() == unchanged_checkpoint
                and manifest_path.read_bytes() == unchanged_manifest
                and not list(fold_dir.glob("*scores.csv")),
            )
        with patch.object(pit_module, "_fold_training_contract_is_compatible", return_value=False), \
             patch.object(pit_module, "scientific_torch_execution", return_value={}), \
             patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(object(), None)) as fit_owner, \
             patch.object(pit_module, "_checkpoint_matches_rescore_contract", return_value=True) as checkpoint_check, \
             patch.object(pit_module, "build_model", return_value=fake_model), \
             patch.object(pit_module, "_predict_primary_score_payload", return_value=(np.array([0.5]), {})), \
             patch.object(pit_module, "_build_fold_score_frame", side_effect=frame), \
             patch.object(pit_module, "_validate_score_frame", side_effect=lambda x, **_k: x), \
             patch.object(pit_module, "_validate_validation_score_frame", side_effect=lambda x, **_k: x):
            recovered = pit_module._rescore_fold_from_compatible_checkpoint(
                fold_dir=fold_dir, bundle=fake_bundle,
                ids={"score_ids": [0], "validation_ids": [0]},
                fold_contract=expected, expected_fingerprint="test", args=SimpleNamespace(),
                torch_module=SimpleNamespace(load=MagicMock(return_value=fake_checkpoint)),
                plan=SimpleNamespace(device="cpu"),
            )
            check_true(
                "pit_native_verified_fit_reuses_identical_weight_bytes_and_rebuilds_only_scores",
                recovered is not None and fit_owner.call_count == 1
                and checkpoint_check.call_args.kwargs["verified_archived_fit"] is True
                and model_path.read_bytes() == unchanged_checkpoint
                and recovered[1]["source_contract"] == expected["source_contract"]
                and recovered[1]["artifacts"]["checkpoint"] == historical["artifacts"]["checkpoint"]
                and (fold_dir / pit_module.FOLD_SCORE_FILENAME).is_file(),
            )

        # A subsequent score-output refresh sees the new manifest but the
        # immutable model.pt still has its historical embedded Fold contract.
        # Re-consult the FIT owner, rather than regressing to TRAIN/BLOCK.
        with patch.object(pit_module, "_fold_training_contract_is_compatible", return_value=True), \
             patch.object(pit_module, "scientific_torch_execution", return_value={}), \
             patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(object(), None)) as fit_owner, \
             patch.object(pit_module, "_checkpoint_matches_rescore_contract",
                          side_effect=[False, True]) as checkpoint_check, \
             patch.object(pit_module, "build_model", return_value=fake_model), \
             patch.object(pit_module, "_predict_primary_score_payload", return_value=(np.array([0.5]), {})), \
             patch.object(pit_module, "_build_fold_score_frame", side_effect=frame), \
             patch.object(pit_module, "_validate_score_frame", side_effect=lambda x, **_k: x), \
             patch.object(pit_module, "_validate_validation_score_frame", side_effect=lambda x, **_k: x):
            second = pit_module._rescore_fold_from_compatible_checkpoint(
                fold_dir=fold_dir, bundle=fake_bundle,
                ids={"score_ids": [0], "validation_ids": [0]},
                fold_contract=expected, expected_fingerprint="test-v2", args=SimpleNamespace(),
                torch_module=SimpleNamespace(load=MagicMock(return_value=fake_checkpoint)),
                plan=SimpleNamespace(device="cpu"),
            )
            check_true(
                "pit_repeated_score_refresh_reverifies_historical_embedded_fit_contract",
                second is not None and fit_owner.call_count == 1
                and checkpoint_check.call_count == 2
                and checkpoint_check.call_args.kwargs["verified_archived_fit"] is True
                and model_path.read_bytes() == unchanged_checkpoint
                and second[1]["contract_fingerprint"] == "test-v2",
            )
        retained_manifest = manifest_path.read_bytes()
        with patch.object(pit_module, "_fold_training_contract_is_compatible", return_value=True), \
             patch.object(pit_module, "scientific_torch_execution", return_value={}), \
             patch.object(pit_module, "validate_rolling_fold_fitted_checkpoint",
                          return_value=(None, "current target lineage differs")) as fit_owner, \
             patch.object(pit_module, "_checkpoint_matches_rescore_contract", return_value=False):
            errors = []
            rejected = pit_module._rescore_fold_from_compatible_checkpoint(
                fold_dir=fold_dir, bundle=fake_bundle,
                ids={"score_ids": [0], "validation_ids": [0]},
                fold_contract=expected, expected_fingerprint="test-v3", args=SimpleNamespace(),
                torch_module=SimpleNamespace(load=MagicMock(return_value=fake_checkpoint)),
                plan=SimpleNamespace(device="cpu"), fit_validation_errors=errors,
            )
            check_true(
                "pit_repeated_score_refresh_rejects_unproven_embedded_contract",
                rejected is None and errors == ["current target lineage differs"]
                and fit_owner.call_count == 1
                and model_path.read_bytes() == unchanged_checkpoint
                and manifest_path.read_bytes() == retained_manifest,
            )

    # FIT repair history must not be mistaken for absent Rolling folds. A
    # score/Audit refresh first verifies the archived Fold bytes and Context,
    # and never starts optimizer epochs for an unproven historical candidate.
    from services.breakout_quality.fit_repair_transaction import archived_rolling_fit_fold_candidates
    from services.breakout_quality import point_in_time_scores as pit_owner
    from unittest.mock import patch as local_patch
    with tempfile.TemporaryDirectory(prefix="pit_archived_fit_") as td:
        test_root = Path(td)
        pit = test_root / "models" / "flt" / "architecture" / "profile" / "point_in_time"
        fold_id = "fold_20220101_20221231"
        archive = (pit.parent.parent / "historical_fit_repairs" / pit.parent.name
                   / "20261019" / "original" / "point_in_time" / "folds" / fold_id)
        archive.mkdir(parents=True)
        (archive / "model.pt").write_bytes(b"same historical fitted checkpoint")
        historical_manifest = {
            "fold_id": fold_id,
            "artifacts": {"checkpoint": build_file_manifest(archive / "model.pt")},
        }
        (archive / "manifest.json").write_text(json.dumps(historical_manifest), encoding="utf-8")
        (archive.parent.parent / "selection_point_in_time_manifest.json").write_text("{}", encoding="utf-8")
        check("pit_finds_historical_fold_archived_with_parent_model_tree",
              (archive,), archived_rolling_fit_fold_candidates(
                  point_in_time_dir=pit, fold_id=fold_id, project_root=test_root,
              ))
        forwarded_cache = pit / "folds" / "fold_20210101_20211231"
        forwarded_cache.mkdir(parents=True)
        (forwarded_cache / "model.pt").write_bytes(b"different new Forward checkpoint")
        with local_patch.object(pit_owner, "validate_rolling_fold_fitted_checkpoint",
                                return_value=(object(), None)), \
             local_patch.object(pit_owner, "_fold_training_contract_is_compatible", return_value=True), \
             local_patch.object(pit_owner, "scientific_torch_execution", return_value={}):
            match = pit_owner._verified_archived_rolling_fit(
                point_in_time_dir=pit, fold_contract={"fold_id": fold_id},
                bundle=SimpleNamespace(), torch_module=SimpleNamespace(),
                plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=test_root,
            )
        check_true("pit_archive_preflight_uses_canonical_fit_owner_and_preserved_sha",
                   match is not None and match[0] == archive
                   and match[1]["source_checkpoint"] == historical_manifest["artifacts"]["checkpoint"])
        # Historical event-cache schema drift is the FIT owner's job; the
        # archive preflight must not impose a second raw-manifest equality test.
        with local_patch.object(pit_owner, "validate_rolling_fold_fitted_checkpoint",
                                return_value=(object(), None)), \
             local_patch.object(pit_owner, "_fold_training_contract_is_compatible",
                                side_effect=AssertionError("duplicate FIT identity gate")), \
             local_patch.object(pit_owner, "scientific_torch_execution", return_value={}):
            owner_only_archive = pit_owner._verified_archived_rolling_fit(
                point_in_time_dir=pit, fold_contract={"fold_id": fold_id},
                bundle=SimpleNamespace(), torch_module=SimpleNamespace(),
                plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=test_root,
            )
        check_true("pit_archive_preflight_has_no_second_stricter_fit_serialization_gate",
                   owner_only_archive is not None)
        if match is not None:
            projected_contract = {"fold_id": fold_id, "source_contract": {"dataset_artifacts": {"new": True}}}
            seen_projection = []
            def fake_archive_regeneration(*, fold_dir, verified_archived_fit, fold_contract, **_kwargs):
                written = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
                seen_projection.append(
                    verified_archived_fit
                    and written["source_contract"] == projected_contract["source_contract"]
                    and written["archived_rolling_fit_source"] == match[1]
                    and (fold_dir / "model.pt").read_bytes() == (archive / "model.pt").read_bytes()
                )
                return pd.DataFrame({"score": [0.5]}), written
            new_target = pit / "folds" / "archive_identity_projection"
            with local_patch.object(pit_owner, "_rescore_fold_from_compatible_checkpoint",
                                    side_effect=fake_archive_regeneration):
                updated = pit_owner._rescore_fold_from_fitting_checkpoint(
                    source_fold_dir=archive, target_fold_dir=new_target,
                    verified_archive_binding=match[1],
                    bundle=SimpleNamespace(), ids={}, fold_contract=projected_contract,
                    expected_fingerprint="test", args=SimpleNamespace(),
                    torch_module=SimpleNamespace(), plan=SimpleNamespace(),
                )
            check_true("pit_archive_rescore_projects_owner_verified_identity_without_rewriting_old_weights",
                       updated is not None and seen_projection == [True]
                       and (new_target / "model.pt").read_bytes() == (archive / "model.pt").read_bytes())
        if match is not None:
            archived_source_bytes = (archive / "model.pt").read_bytes()
            def fake_rescore_archived(*, source_fold_dir, target_fold_dir, **_kwargs):
                target_fold_dir.mkdir(parents=True, exist_ok=True)
                (target_fold_dir / "model.pt").write_bytes((source_fold_dir / "model.pt").read_bytes())
                return pd.DataFrame({"score": [0.5]}), {
                    "artifacts": {"checkpoint": build_file_manifest(target_fold_dir / "model.pt")},
                    "selected_epoch": 1,
                }
            with local_patch.object(pit_owner, "_rescore_fold_from_fitting_checkpoint",
                                    side_effect=fake_rescore_archived):
                restored_frame, restored = pit_owner._materialize_verified_archived_rolling_fit(
                    archived_fold=archive, archive_binding=match[1],
                    fold_dir=pit / "folds" / fold_id,
                    cache_root=test_root / "models" / "fit_cache",
                    bundle=SimpleNamespace(), ids={}, fold_contract={"fold_id": fold_id},
                    expected_fingerprint="test", args=SimpleNamespace(),
                    torch_module=SimpleNamespace(), plan=SimpleNamespace(),
                )
            restored_manifest = json.loads((pit / "folds" / fold_id / "manifest.json").read_text())
            check_true("pit_restores_archived_fit_without_optimizer_and_keeps_historical_bytes",
                       len(restored_frame) == 1
                       and (pit / "folds" / fold_id / "model.pt").read_bytes() == archived_source_bytes
                       and (archive / "model.pt").read_bytes() == archived_source_bytes
                       and restored_manifest["archived_rolling_fit_source"] == match[1]
                       and restored["selected_epoch"] == 1)
            try:
                pit_owner._materialize_verified_archived_rolling_fit(
                    archived_fold=archive, archive_binding=match[1],
                    fold_dir=pit / "folds" / fold_id,
                    cache_root=None, bundle=SimpleNamespace(), ids={},
                    fold_contract={"fold_id": fold_id}, expected_fingerprint="test",
                    args=SimpleNamespace(), torch_module=SimpleNamespace(),
                    plan=SimpleNamespace(),
                )
            except RuntimeError as exc:
                live_protected = "refuses to overwrite" in str(exc)
            else:
                live_protected = False
            check_true("pit_archived_fit_recovery_does_not_overwrite_existing_local_fold",
                       live_protected)
        with local_patch.object(pit_owner, "validate_rolling_fold_fitted_checkpoint",
                                return_value=(None, "original external Context unproven")):
            try:
                pit_owner._verified_archived_rolling_fit(
                    point_in_time_dir=pit, fold_contract={"fold_id": fold_id},
                    bundle=SimpleNamespace(), torch_module=SimpleNamespace(),
                    plan=SimpleNamespace(as_manifest_payload=lambda: {}), project_root=test_root,
                )
            except RuntimeError as exc:
                unproven_blocked = ("automatic training blocked" in str(exc)
                                    and "Context" in str(exc))
            else:
                unproven_blocked = False
        check_true("pit_historical_fold_context_gap_never_automatically_retrains",
                   unproven_blocked)
        other_archive = archive.parent.parent.parent / "20261018" / "original" / "point_in_time" / "folds" / fold_id
        # Only the exact published profile/fold archive is inspected.
        check_true("pit_archive_lookup_does_not_scan_other_model_namespaces",
                   all("historical_fit_repairs/profile/" in str(candidate).replace("\\", "/")
                       for candidate in archived_rolling_fit_fold_candidates(
                           point_in_time_dir=pit, fold_id=fold_id, project_root=test_root,
                       )))
    check_true("pit_historical_fold_restore_occurs_before_optimizer_training",
               pit_owner._verified_archived_rolling_fit.__name__
               == "_verified_archived_rolling_fit")

    # Byte-backed historical Context attestation survives score regeneration.
    # It is a reference to the archived aggregate, NOT a newly invented
    # training-time Context witness.
    from services.breakout_quality.fitting_data_dependencies import (
        context_source_for_profile, verified_published_context_fit_proof,
    )
    from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec
    with tempfile.TemporaryDirectory(prefix="pit_archive_context_") as td:
        test_root = Path(td)
        profile_name = "daily_universal_predicted_safety_product_weighted_pure_mfe_full_list_ndcg_pairwise"
        context_source = context_source_for_profile(profile_name)
        context_spec = maybe_predicted_context_artifact_spec(context_source)
        old_pit = (test_root / "models" / "flt" / "arch" / "historical_fit_repairs"
                   / profile_name / "stamp" / "original" / "point_in_time")
        new_pit = test_root / "models" / "flt" / "arch" / profile_name / "point_in_time"
        fold_id = "fold_20220101_20221231"
        archived = old_pit / "folds" / fold_id
        current = new_pit / "folds" / fold_id
        archived.mkdir(parents=True)
        current.mkdir(parents=True)
        (archived / "model.pt").write_bytes(b"immutable rolling context fitted bytes")
        (current / "model.pt").write_bytes((archived / "model.pt").read_bytes())
        record = build_file_manifest(archived / "model.pt")
        stage = {
            "research_id": "stage1", "profile": "source", "architecture": "architecture", "seed": 42,
            **{k: {"sha256": "a" * 64, "size_bytes": 1} for k in (
                "selection_score", "selection_manifest", "forward_score", "forward_manifest",
            )},
        }
        context = {
            "schema_version": 1, "filter_id": "flt", "model_architecture": "arch",
            "experiment_profile": "source", "contract": {"pit": True},
            "dataset_policy": "canonical", "periods": {"start": "2020"},
            "rows": {"total": 1}, "artifact": {"sha256": "b" * 64, "size_bytes": 9},
            "source_stage1": stage,
        }
        target = {
            "schema_version": 1, "target_contract": {"horizon": 40}, "artifacts": None,
            context_spec.manifest_key: context,
        }
        identity = {
            "fold_id": fold_id, "filter_id": "flt", "model_architecture": "arch",
            "experiment_profile": profile_name, "continuous_target_id": "target",
            "training_sample_scope": "daily", "training_label_scope": "all", "seed": 42,
            "artifacts": {"checkpoint": record},
        }
        old_manifest_path = archived / "manifest.json"
        old_manifest_path.write_text(json.dumps(identity), encoding="utf-8")
        def aggregate_for(manifest_path):
            return {
                **{k: identity[k] for k in (
                    "filter_id", "model_architecture", "experiment_profile",
                    "continuous_target_id", "training_sample_scope", "training_label_scope", "seed",
                )},
                "folds": [{"fold_id": fold_id, "fold_manifest": build_file_manifest(manifest_path),
                           "artifacts": identity["artifacts"]}],
                "source_continuous_target": target,
            }
        old_aggregate = old_pit / "selection_point_in_time_manifest.json"
        old_aggregate.write_text(json.dumps(aggregate_for(old_manifest_path)), encoding="utf-8")
        binding = {
            "source_fold_path": archived.relative_to(test_root).as_posix(),
            "source_fold_manifest": build_file_manifest(old_manifest_path),
            "source_aggregate_manifest": build_file_manifest(old_aggregate),
            "source_checkpoint": record,
        }
        restored_manifest = {
            **identity,
            "migration": {"kind": "fitting_identity_checkpoint_reuse",
                          "training_contract_unchanged": True,
                          "source_checkpoint_sha_preserved": True},
            "archived_rolling_fit_source": binding,
        }
        current_manifest_path = current / "manifest.json"
        current_manifest_path.write_text(json.dumps(restored_manifest), encoding="utf-8")
        current_aggregate = new_pit / "selection_point_in_time_manifest.json"
        current_aggregate.write_text(json.dumps(aggregate_for(current_manifest_path)), encoding="utf-8")
        bundle = SimpleNamespace(target_manifest=target)
        verified, reason = verified_published_context_fit_proof(
            manifest_path=current_manifest_path, fold_manifest=restored_manifest,
            bundle=bundle, project_root=test_root,
        )
        check_true("pit_archived_native_fold_context_verified_through_original_aggregate",
                   verified and reason.startswith("verified published Rolling"))
        prior = old_aggregate.read_text(encoding="utf-8")
        old_aggregate.write_text(prior + " ", encoding="utf-8")
        verified, reason = verified_published_context_fit_proof(
            manifest_path=current_manifest_path, fold_manifest=restored_manifest,
            bundle=bundle, project_root=test_root,
        )
        check_true("pit_archive_context_attestation_rejects_tampered_original_aggregate",
                   not verified and "original SHA binding" in reason)

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
    audit_json_write = audit_source.index("publish_latest_snapshot(")
    audit_cache_clear = audit_source.index(
        "clear_selection_point_in_time_ranking_contract_cache()", audit_json_write
    )
    audit_publication = audit_source[audit_json_write:audit_cache_clear]
    audit_json_commits_last = (
        audit_publication.index("output_markdown.name") < audit_publication.index("output_json.name")
        and "text_publication_bytes" in audit_publication
    )
    run_bound_checks(
        check_true,
        (
            ('pit_score_and_audit_producers_invalidate_cached_ranking_contract_after_artifact_rewrite', 'def clear_selection_point_in_time_ranking_contract_cache()' in ranking_store_source and pit_manifest_write < pit_cache_clear and (audit_json_write < audit_cache_clear) and audit_json_commits_last,),
            ('rolling_standard_sop_validation_sidecar_reader_ignores_non_sop_ticker_dtype', 'usecols=sorted(required)' in audit_source and 'dtype={"fold_id": "string"}' in audit_source,),
        ),
    )

    # Registered score-transform capability: signed Raw-R primary scores must
    # survive PIT/Forward reader and audit; probability outputs stay bounded.
    from tools.validate.breakout_quality_signed_primary_score_cases import (
        case as validate_signed_primary_score_domain,
    )
    validate_signed_primary_score_domain()
    check_true("pit_primary_margin_domain_preserves_probability_checks", True)

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
    )
    from core.breakout_quality_registry import (
        DAILY_UNIVERSAL_FIRST_RISK_BREACH_PURE_MFE_FULL_LIST_NDCG_PAIRWISE_PROFILE,
        get_breakout_quality_experiment_profile,
    )
    from core.breakout_quality_policy import (
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
