"""Mutation-tested Research FIT lifecycle and architecture invariants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .checks import bind_synthetic_case
from .research_lifecycle_architecture import (
    FIT_OWNER, PIT_PRODUCER, FORWARD_PRODUCER, RESEARCH_APPLICATION,
    check_research_fit_architecture_sources,
    BENCHMARK_OWNER, STRATEGY_SINGLE, STRATEGY_MULTI,
    check_strategy_benchmark_architecture_sources, check_research_fit_architecture,
    MODEL_SCORE_OWNER, FORWARD_MODEL_SOURCE_OWNER,
    check_model_forward_source_architecture_sources,
)


def validate_research_fit_lifecycle_architecture_case(_base_params):
    case_id = "RESEARCH_FIT_LIFECYCLE_ARCHITECTURE"
    results, summary, check, check_true = bind_synthetic_case(
        case_id, "meta_architecture_contract",
    )
    root = Path(__file__).resolve().parents[2]
    sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in (FIT_OWNER, PIT_PRODUCER, FORWARD_PRODUCER, RESEARCH_APPLICATION)
    }
    check("canonical_source_owner_and_dependency_boundary", (),
          check_research_fit_architecture_sources(sources))
    forward_sources = {
        **sources,
        MODEL_SCORE_OWNER: (root / MODEL_SCORE_OWNER).read_text(encoding="utf-8"),
        FORWARD_MODEL_SOURCE_OWNER: (root / FORWARD_MODEL_SOURCE_OWNER).read_text(encoding="utf-8"),
    }
    check("forward_single_multiseed_canonical_owner_boundary", (),
          check_model_forward_source_architecture_sources(forward_sources))
    shadow = dict(forward_sources)
    shadow[RESEARCH_APPLICATION] = shadow[RESEARCH_APPLICATION].replace(
        "contract, reason = load_forward_model_source(",
        "contract, reason = load_continuous_ranker_oos_contract(", 1,
    )
    check_true("forward_robustness_cannot_bypass_source_owner",
               bool(check_model_forward_source_architecture_sources(shadow)))
    shadow = dict(forward_sources)
    shadow[FORWARD_MODEL_SOURCE_OWNER] = shadow[FORWARD_MODEL_SOURCE_OWNER].replace(
        "contract = load_continuous_ranker_oos_contract_at_paths(",
        "contract = load_continuous_ranker_oos_contract(", 1,
    )
    check_true("forward_isolated_seed_must_use_canonical_path_aware_validator",
               bool(check_model_forward_source_architecture_sources(shadow)))
    shadow = dict(forward_sources)
    shadow[RESEARCH_APPLICATION] += (
        "\ndef shadow_forward_ready(model_path):\n"
        "    return (model_path / 'model.pt').is_file()\n"
    )
    # Duplicate readiness must be in a required adapter, not an unrelated helper:
    # the existing FIT architecture guard checks renamed generic file-based helpers.
    check_true("forward_app_cannot_recreate_presence_check",
               bool(check_research_fit_architecture_sources(shadow)))

    # Byte-backed Forward fixture: seed paths must be parsed by the *real*
    # complete Forward manifest/report validator. Only external current-upstream
    # availability and scientific SOP metric calculations are isolated.
    import json
    from core.breakout_quality_policy import get_breakout_quality_model_research_settings
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from filters.breakout_quality import ranking_score_store as forward_store
    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.paths import build_filter_artifact_paths_from_dir
    from filters.breakout_quality.ranker_training_contract import training_semantics
    from services.breakout_quality import forward_model_source as forward_source

    model_settings = get_breakout_quality_model_research_settings()
    profile = get_breakout_quality_experiment_profile(str(model_settings.experiment_profile))
    semantic = training_semantics(profile)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    with TemporaryDirectory() as td:
        project = Path(td)
        isolated_model_dir = project / "model_seed_101"
        isolated_report_dir = project / "reports_seed_101"
        isolated_model_dir.mkdir()
        isolated_report_dir.mkdir()
        paths = build_filter_artifact_paths_from_dir(
            filter_id=model_settings.filter_id,
            model_architecture=model_settings.model_architecture,
            experiment_profile=profile.name, model_dir=isolated_model_dir,
        )
        paths.model_path.write_bytes(b"synthetic_forward_101")
        report_path = isolated_report_dir / forward_store.CONTINUOUS_RANKER_REPORT_FILENAME
        coverage = {
            "inference_eligible_groups": 12,
            "target_evaluable_groups": 8,
            "future_target_required_for_score": False,
        }
        eligibility = forward_store.build_score_eligibility_contract(profile)
        manifest = {
            "filter_id": str(model_settings.filter_id),
            "model_architecture": str(model_settings.model_architecture),
            "experiment_profile": profile.name,
            "training_objective": profile.training_objective,
            "training_sample_scope": profile.training_sample_scope,
            "training_label_scope": profile.training_label_scope,
            "training_semantics": semantic,
            "continuous_target_id": profile.continuous_target_id,
            "score_eligibility_contract": eligibility,
            "forward_score_coverage": coverage,
            "model": build_file_manifest(paths.model_path),
            "outer_oos_policy": {"oos_start_date": "2021-01-01", "configured_oos_end_date": "2021-12-31"},
            "model_information_cutoff": "2020-12-31",
        }
        report = {
            "filter_id": str(model_settings.filter_id),
            "model_architecture": str(model_settings.model_architecture),
            "experiment_profile": profile.name,
            "experiment_settings": profile.as_manifest_payload(),
            "training": {
                "objective": profile.training_objective,
                "loss": profile.loss_name,
                "sample_scope": profile.training_sample_scope,
                "training_label_scope": profile.training_label_scope,
                **semantic,
                "pairwise_reduction": recipe.pairwise_reduction,
                "pair_weight_policy": recipe.objective_policy.pair_weight_policy,
                "epoch_selection_metric": profile.epoch_selection_metric,
                "seed": 101,
            },
            "score_eligibility_contract": eligibility,
            "forward_score_coverage": coverage,
            "status": "RESULT_AVAILABLE",
            "standard_model_sop": {"evaluation_mode": "forward_oos"},
        }
        def publish():
            paths.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path.write_text(json.dumps(report), encoding="utf-8")

        def isolated_source(seed=101, **overrides):
            return forward_source.load_forward_model_source(
                project_root=project, filter_id=str(model_settings.filter_id),
                model_architecture=str(model_settings.model_architecture),
                experiment_profile=profile.name, expected_seed=seed,
                dataset="full", max_tickers=0,
                model_dir_override=isolated_model_dir,
                research_output_dir_override=isolated_report_dir,
                **overrides,
            )

        publish()
        with patch.object(forward_source, "fitted_model_settings_issues", return_value=()), \
             patch.object(forward_source, "_forward_source_provenance_issues", return_value=()), \
             patch.object(forward_source, "standard_model_sop_completeness_issues", return_value=()):
            source, reason = isolated_source()
            check_true("forward_isolated_full_canonical_contract_accepts_verified_seed_without_csv",
                       source is not None and reason is None and source.seed == 101
                       and source.report_path == report_path)
            check_true("forward_isolated_source_path_is_not_canonical_publication",
                       source.manifest_path == paths.manifest_path)
            wrong, wrong_reason = isolated_source(seed=202)
            check_true("forward_robustness_wrong_seed_cannot_reuse",
                       wrong is None and "seed" in str(wrong_reason))
            check_true("forward_isolated_requires_atomic_model_report_path_pair",
                       forward_source.load_forward_model_source(
                           project_root=project, filter_id=str(model_settings.filter_id),
                           model_architecture=str(model_settings.model_architecture),
                           experiment_profile=profile.name, expected_seed=101,
                           dataset="full", max_tickers=0, model_dir_override=isolated_model_dir,
                       )[0] is None)
            # There is no lru-cache for seed-isolated artifacts: a post-check
            # model mutation must invalidate the immediately following reuse.
            paths.model_path.write_bytes(b"tampered-forward-model")
            corrupt, corrupt_reason = isolated_source()
            check_true("forward_robustness_checkpoint_sha_changed_blocks_reuse",
                       corrupt is None and "SHA256" in str(corrupt_reason))
            paths.model_path.write_bytes(b"synthetic_forward_101")
            manifest["forward_score_coverage"] = dict(coverage, target_evaluable_groups=99)
            publish()
            invalid_coverage, coverage_reason = isolated_source()
            check_true("forward_robustness_malformed_score_coverage_fails_closed",
                       invalid_coverage is None and "coverage" in str(coverage_reason))
            manifest["forward_score_coverage"] = coverage
            manifest["training_semantics"] = {"batching": "illegally_changed"}
            publish()
            invalid_semantics, semantic_reason = isolated_source()
            check_true("forward_robustness_training_semantics_drift_fails_closed",
                       invalid_semantics is None and "training semantics" in str(semantic_reason))
            manifest["training_semantics"] = semantic
            report["training"]["seed"] = 202
            publish()
            wrong_report_seed, report_seed_reason = isolated_source()
            check_true("forward_robustness_report_seed_mismatch_fails_closed",
                       wrong_report_seed is None and "seed" in str(report_seed_reason))
            report["training"]["seed"] = 101
            report["standard_model_sop"]["evaluation_mode"] = "rolling_oos"
            publish()
            incorrect_mode, mode_reason = isolated_source()
            check_true("forward_robustness_wrong_evaluation_mode_fails_closed",
                       incorrect_mode is None and "evaluation_mode" in str(mode_reason))
            report["standard_model_sop"]["evaluation_mode"] = "forward_oos"
            report["training"].pop("seed")
            publish()
            missing_seed, missing_seed_reason = isolated_source()
            check_true("forward_robustness_missing_training_seed_fails_closed",
                       missing_seed is None and "seed" in str(missing_seed_reason))
            report["training"]["seed"] = 101
            publish()
            with patch.object(forward_source, "standard_model_sop_completeness_issues",
                              return_value=("M3:oos:missing evidence",)):
                missing_sop, sop_reason = isolated_source()
            check_true("forward_robustness_missing_standard_sop_cannot_reuse",
                       missing_sop is None and "SOP" in str(sop_reason))
            with patch.object(forward_source, "_forward_source_provenance_issues",
                              return_value=("Dataset: source lineage stale",)):
                stale_source, source_reason = isolated_source()
            check_true("forward_robustness_stale_source_lineage_cannot_reuse",
                       stale_source is None and "lineage" in str(source_reason))
            valid_again, reason_again = isolated_source()
            check_true("forward_robustness_valid_source_recovers_without_refit",
                       valid_again is not None and reason_again is None)

    reversed_dependency = dict(sources)
    reversed_dependency[FIT_OWNER] += (
        "\ndef shadow_pit_loader():\n"
        "    from services.breakout_quality.point_in_time_scores import parse_args\n"
        "    return parse_args([])\n"
    )
    check_true("lazy_fit_owner_to_pit_import_blocked",
               any("reverse dependency" in issue for issue in
                   check_research_fit_architecture_sources(reversed_dependency)))
    missing_proof_binding = dict(sources)
    missing_proof_binding[FORWARD_PRODUCER] = missing_proof_binding[FORWARD_PRODUCER].replace(
        "rolling_fold_proof_factory=build_rolling_first_fold_fit_proof,",
        "rolling_fold_proof_factory=lambda **kwargs: (None, None),", 1,
    )
    check_true("forward_cannot_inject_unverified_fold_proof",
               any("PIT fold proof binding" in issue for issue in
                   check_research_fit_architecture_sources(missing_proof_binding)))

    # Mutation tests deliberately introduce new business logic under *different*
    # names to catch drift not detected by a list of old legacy function labels.
    mut = dict(sources)
    mut[PIT_PRODUCER] += "\ndef fold_training_identity(contract):\n    return 'shadow'\n"
    check_true("second_pit_fit_identity_blocked", bool(check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[PIT_PRODUCER] += ("\ndef load_forward_compatible_rolling_checkpoint(*args, **kwargs):\n"
                          "    return None, 'shadow'\n")
    check_true("retired_pit_fit_bridge_cannot_return",
               any("retired PIT" in issue for issue in check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[RESEARCH_APPLICATION] += (
        "\ndef _has_continuous_forward_fitted_candidate(settings):\n"
        "    return True\n"
    )
    check_true("retired_application_fit_readiness_cannot_return",
               any("retired application" in issue for issue in check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[FIT_OWNER] = mut[FIT_OWNER].replace(
        "rolling, rolling_reason = resolve_compatible_rolling_first_fold(",
        "rolling, rolling_reason = nonexistent_shadow_resolver(", 1,
    )
    check_true("owner_cross_mode_bypass_blocked",
               bool(check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[RESEARCH_APPLICATION] += (
        "\ndef new_shadow_fit_ready(path):\n"
        "    return (path / 'model.pt').is_file()\n"
    )
    check_true("renamed_app_file_based_fit_ready_blocked", bool(check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[FORWARD_PRODUCER] += (
        "\ndef new_forward_fallback(path):\n"
        "    return load_reusable_fitted_model_contract(path)\n"
    )
    check_true("new_forward_direct_fitted_loader_blocked", bool(check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[FORWARD_PRODUCER] += (
        "\nfrom services.breakout_quality.fitted_model_artifacts import "
        "load_reusable_fitted_model_contract as secret_model_loader\n"
        "def hidden_alias_fallback(path):\n"
        "    return secret_model_loader(path)\n"
    )
    check_true("aliased_second_forward_fitted_loader_blocked",
               bool(check_research_fit_architecture_sources(mut)))
    mut = dict(sources)
    mut[FIT_OWNER] = mut[FIT_OWNER].replace(
        "def select_unique_fitted_contract(", "def removed_unique_fitted_contract(", 1,
    )
    check_true("missing_owner_conflict_guard_blocked", bool(check_research_fit_architecture_sources(mut)))

    benchmark_sources = {
        name: (root / name).read_text(encoding="utf-8")
        for name in (BENCHMARK_OWNER, STRATEGY_SINGLE, STRATEGY_MULTI)
    }
    check("canonical_strategy_0050_owner_boundary", (),
          check_strategy_benchmark_architecture_sources(benchmark_sources))
    check("all_research_owner_boundaries", (),
          check_research_fit_architecture(root))
    mut = dict(benchmark_sources)
    mut[STRATEGY_MULTI] += (
        "\ndef _robustness_benchmark_yearly(results):\n"
        "    return results[0]['benchmark_yearly']\n"
    )
    check_true("second_multi_seed_0050_selector_blocked",
               bool(check_strategy_benchmark_architecture_sources(mut)))
    mut = dict(benchmark_sources)
    mut[STRATEGY_SINGLE] += (
        "\ndef shadow_benchmark_yearly(prices, dates):\n"
        "    return _build_benchmark_period_stats(prices, dates)\n"
    )
    check_true("second_single_seed_benchmark_producer_blocked",
               bool(check_strategy_benchmark_architecture_sources(mut)))
    mut = dict(benchmark_sources)
    mut[BENCHMARK_OWNER] = mut[BENCHMARK_OWNER].replace(
        "def resolve_strategy_robustness_report_benchmark_yearly(",
        "def disabled_resolve_strategy_robustness_report_benchmark_yearly(", 1,
    )
    check_true("missing_benchmark_canonical_owner_blocked",
               bool(check_strategy_benchmark_architecture_sources(mut)))
    from services.research.strategy_compare_benchmark import (
        collect_consistent_benchmark_yearly,
        collect_strategy_pair_benchmark_yearly,
        resolve_strategy_robustness_report_benchmark_yearly,
    )
    example = [{"year": 2021, "year_return_pct": 8.25}]
    check("single_seed_yearly_evidence_canonical", example,
          collect_strategy_pair_benchmark_yearly({
              "C96": {"payload": {"benchmark_yearly": example}},
              "C97": {"payload": {"benchmark_yearly": example}},
          }))
    check("fixed_multi_seed_yearly_evidence_canonical", example,
          resolve_strategy_robustness_report_benchmark_yearly(
              project_root=root, contract={}, fixed_results={
                  "C96": {"benchmark_yearly": example},
                  "C97": {"benchmark_yearly": example},
              }, seed_yearly_frame=None,
          ))
    try:
        collect_consistent_benchmark_yearly({
            "C96": {"benchmark_yearly": example},
            "C97": {"benchmark_yearly": [{"year": 2021, "year_return_pct": 99.0}]},
        })
        differing_benchmark_blocked = False
    except ValueError:
        differing_benchmark_blocked = True
    check_true("divergent_0050_yearly_evidence_blocked", differing_benchmark_blocked)

    from services.breakout_quality import fitted_model_artifacts as owner
    from services.breakout_quality import point_in_time_scores as pit
    check_true("retired_pit_rolling_fit_resolver_removed",
               not hasattr(pit, "load_forward_compatible_rolling_checkpoint"))
    # Fold identity serialization MUST match the historical byte-level contract.
    identity = {
        "filter_id": "synthetic", "model_architecture": "synthetic",
        "experiment_profile": "profile", "continuous_target_id": "target",
        "training_label_scope": "all", "training_sample_scope": "daily",
        "seed": 42, "model_information_cutoff": "2020-12-31",
        "model_spec": {"kind": "stub"}, "experiment_settings": {"x": 1},
        "training_settings": {"epochs_max": 3, "learning_rate": 0.001},
        "source_contract": {"dataset_policy": "synthetic"},
        "lookahead_contract": {"oos_used_for_training_or_epoch_selection": False},
        "planned_periods": {"validation_start": "2019-01-01", "validation_end": "2020-12-31", "score_start": "2021-01-01"},
        "observed_periods": {"inner_train": {}, "validation": {}, "final_refit": {}},
        "group_counts": {"inner_train": 100, "validation": 20, "final_refit": 120},
        "event_row_counts": {"inner_train": 0, "validation": 0, "final_refit": 0},
    }
    a = owner.fold_training_identity(identity)
    check("pit_import_has_same_owner_identity", a, pit.fold_training_identity(identity))
    with_score_end = deepcopy(identity)
    with_score_end["planned_periods"]["score_end"] = "2021-12-31"
    check("evaluation_end_does_not_change_fit_key", a, owner.fold_training_identity(with_score_end))
    changed_seed = deepcopy(identity)
    changed_seed["seed"] = 43
    check_true("changed_seed_requires_different_fit", a != owner.fold_training_identity(changed_seed))
    changed_train = deepcopy(identity)
    changed_train["group_counts"]["inner_train"] = 99
    check_true("changed_training_cohort_requires_different_fit", a != owner.fold_training_identity(changed_train))
    check_true("native_same_contract_accepts", owner._fold_training_contract_is_compatible(
        identity, expected_contract=identity,
    ))
    check_true("native_changed_cohort_rejects", not owner._fold_training_contract_is_compatible(
        changed_train, expected_contract=identity,
    ))

    # FIT owner, not Forward mode, handles native/rolling fallback and conflict.
    with TemporaryDirectory() as td:
        model_dir = Path(td) / "model"
        model_dir.mkdir()
        kwargs = dict(
            reuse_fitted=True, model_dir=model_dir, output_dir=Path(td),
            final_manifest_path=model_dir / "manifest.json",
            report_path=Path(td) / "report.json",
            expected_identity={"fit": "identity"}, project_root=Path(td),
            bundle=SimpleNamespace(), forward_split=SimpleNamespace(),
            forward_args=SimpleNamespace(), forward_plan=SimpleNamespace(),
            torch_module=SimpleNamespace(),
            rolling_fold_proof_factory=pit.build_rolling_first_fold_fit_proof,
        )
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "local missing")), patch.object(
            owner, "resolve_compatible_rolling_first_fold", return_value=(None, "rolling missing"),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("missing_fit_action", "FIT_REQUIRED", result.source)
        check_true("missing_fit_has_evidence_reason", "rolling missing" in str(result.reason))
        (model_dir / "model.pt").write_bytes(b"unverified bytes")
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "stale")), patch.object(
            owner, "resolve_compatible_rolling_first_fold", return_value=(None, "not equivalent"),
        ):
            try:
                owner.resolve_forward_fitted_checkpoint(**kwargs)
                blocked = False
            except RuntimeError as exc:
                blocked = "not equivalent" in str(exc)
        check_true("unverified_existing_fit_blocks_refit", blocked)
        candidate = SimpleNamespace(selected_epoch=3, source="verified_fit")
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(candidate, None)), patch.object(
            owner, "resolve_compatible_rolling_first_fold", side_effect=AssertionError("must not scan Rolling when Forward FIT valid"),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("valid_forward_fit_reuse", "FORWARD_FIT", result.source)
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "old")), patch.object(
            owner, "resolve_compatible_rolling_first_fold", return_value=(candidate, "verified"),
        ), patch.object(owner, "load_model_from_fitted_checkpoint", return_value="model"), patch.object(
            owner, "import_verified_rolling_forward_checkpoint", return_value=(None, True),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("cross_mode_fit_reuse", "ROLLING_FIT", result.source)
        check("cross_mode_fit_loaded_without_refit", "model", result.preloaded_model)

    # Exercise the *migrated* resolver, not only mocked Forward policy calls.
    # PIT produces train/validation cohort facts; the FIT owner alone locates
    # and approves a candidate.  This must work without any one-off MR identity.
    import json
    import numpy as np
    from core.breakout_quality_registry import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
    with TemporaryDirectory() as td:
        folds_dir = Path(td) / "folds"
        fold_dir = folds_dir / "fold_20210101_20211231"
        fold_dir.mkdir(parents=True)
        manifest = {"fold_id": fold_dir.name, "planned_periods": {
            "score_start": "2021-01-01", "score_end": "2021-12-31"}}
        (fold_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (fold_dir / "model.pt").write_bytes(b"synthetic FIT bytes")
        candidate = owner.FittedModelContract(
            manifest_path=fold_dir / "manifest.json", model_path=fold_dir / "model.pt",
            manifest=manifest, fitting_identity={"fixture": "same_fit"},
            selected_epoch=3, epoch_selection={}, final_refit_history=[], source="synthetic",
        )
        args = SimpleNamespace(
            filter_id="fixture", model_architecture="fixture", experiment_profile="fixture",
            seed=42, inner_validation_months=24, use_inner_validation=True,
            epochs=3, batch_size=64, lr=0.001, weight_decay=0.0,
            gradient_clip_norm=1.0, early_stopping_patience=1,
            early_stopping_min_delta=0.0,
        )
        pit_args = SimpleNamespace(**{k: v for k, v in vars(args).items()
                                      if k != "use_inner_validation"}, train_window_months=None)
        bundle = SimpleNamespace(profile=SimpleNamespace(
            training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS))
        split = SimpleNamespace(report={"oos_start_date": "2021-01-01"},
                                inner_train_ids=np.array([1, 2]),
                                validation_ids=np.array([3]),
                                selection_ids=np.array([1, 2, 3]))
        cohort = {"train_ids": np.array([1, 2]), "validation_ids": np.array([3]),
                  "final_ids": np.array([1, 2, 3])}
        call = dict(bundle=bundle, forward_split=split, forward_args=args,
                    forward_plan=SimpleNamespace(), torch_module=SimpleNamespace(),
                    project_root=Path(td),
                    rolling_fold_proof_factory=pit.build_rolling_first_fold_fit_proof)
        with patch.object(owner, "resolve_rolling_fitted_folds_root", return_value=folds_dir), \
             patch.object(pit, "parse_args", return_value=pit_args), \
             patch.object(pit, "_fold_group_ids", return_value=cohort) as fold_ids, \
             patch.object(pit, "_fold_contract_payload", return_value={"fixture": "same_fit"}), \
             patch.object(owner, "scientific_torch_execution", return_value={}), \
             patch.object(owner, "validate_rolling_fold_fitted_checkpoint", return_value=(candidate, None)) as verify:
            found, reason = owner.resolve_compatible_rolling_first_fold(**call)
            check_true("migrated_fit_owner_accepts_matching_first_fold",
                       found == candidate and "verified" in reason)
            check_true("pit_only_supplies_native_cohort_for_fit_owner",
                       fold_ids.call_count == 1 and verify.call_count == 1)
            with patch.object(pit, "_fold_group_ids", return_value={
                **cohort, "train_ids": np.array([7, 8])}):
                mismatch, reason = owner.resolve_compatible_rolling_first_fold(**call)
            check_true("migrated_fit_owner_rejects_cohort_mismatch",
                       mismatch is None and "sample membership" in reason)
            (fold_dir / "manifest.json").unlink()
            try:
                owner.resolve_compatible_rolling_first_fold(**call)
            except RuntimeError as exc:
                missing_manifest_blocked = "missing its fitting manifest" in str(exc)
            else:
                missing_manifest_blocked = False
            check_true("migrated_fit_owner_blocks_corrupt_first_fold",
                       missing_manifest_blocked)

    # No approved science/report identity may change merely because this is a
    # code-organization refactor; those fingerprints live in the report registry.
    return results, summary


def validate_research_model_readiness_architecture_case(_base_params):
    """Cross-mode Stage Readiness and anti-divergence mutation contract."""
    from .research_lifecycle_architecture import (
        MODEL_EVIDENCE_OWNER, MODEL_STAGE_OWNER,
        check_model_stage_readiness_architecture_sources,
    )
    from services.breakout_quality.model_artifact_readiness import (
        resolve_model_artifact_readiness,
    )
    case_id = "RESEARCH_MODEL_STAGE_READINESS_ARCHITECTURE"
    results, summary, check, check_true = bind_synthetic_case(
        case_id, "meta_architecture_contract",
    )
    root = Path(__file__).resolve().parents[2]
    files = (MODEL_EVIDENCE_OWNER, MODEL_STAGE_OWNER, RESEARCH_APPLICATION,
             FORWARD_PRODUCER, PIT_PRODUCER)
    sources = {path: (root / path).read_text(encoding="utf-8") for path in files}
    check("shared_model_readiness_architecture", (),
          check_model_stage_readiness_architecture_sources(sources))
    from .research_lifecycle_architecture import (
        check_model_standard_comparison_audit_boundary_sources,
    )
    check("standard_model_comparison_excludes_one_off_audits", (),
          check_model_standard_comparison_audit_boundary_sources(sources))
    illegal = dict(sources)
    illegal[RESEARCH_APPLICATION] = illegal[RESEARCH_APPLICATION].replace(
        '    console = _render_standard_model_comparison(completed, target="console")',
        "    completed = attach_joint_mfe_d40_extension(completed)\n"
        '    console = _render_standard_model_comparison(completed, target="console")',
        1,
    )
    check_true("standard_comparison_cannot_reintroduce_one_off_audit_call",
               bool(check_model_standard_comparison_audit_boundary_sources(illegal)))
    illegal_import = dict(sources)
    illegal_import[RESEARCH_APPLICATION] += (
        "\nfrom services.breakout_quality.u40_paired_comparison "
        "import attach_paired_u40_extension as hidden_audit\n"
    )
    check_true("one_off_audit_import_cannot_reenter_research_application",
               bool(check_model_standard_comparison_audit_boundary_sources(illegal_import)))
    mutations = (
        (RESEARCH_APPLICATION,
         "\ndef _standard_model_sop_completeness_issues(payload):\n    return ()\n"),
        (RESEARCH_APPLICATION,
         "\ndef hidden_m4_checker(payload):\n    return payload['joint_quality_evaluation_id'] and payload['mfe_pct_mean']\n"),
        (RESEARCH_APPLICATION,
         "\ndef second_report_ready(path):\n    return path.exists() and 'standard_model_report.json' and 'report_contract_fingerprint'\n"),
        (FORWARD_PRODUCER,
         "\ndef standard_model_sop_completeness_issues(payload):\n    return ()\n"),
    )
    for index, (path, extra) in enumerate(mutations):
        edited = dict(sources)
        edited[path] += extra
        check_true(f"non_owner_readiness_mutation_{index}_blocked",
                   bool(check_model_stage_readiness_architecture_sources(edited)))
    edited = dict(sources)
    edited[MODEL_STAGE_OWNER] = edited[MODEL_STAGE_OWNER].replace(
        "def resolve_model_artifact_readiness(",
        "def disabled_model_artifact_readiness(", 1,
    )
    check_true("missing_stage_owner_blocked",
               bool(check_model_stage_readiness_architecture_sources(edited)))
    edited = dict(sources)
    edited[MODEL_STAGE_OWNER] = edited[MODEL_STAGE_OWNER].replace(
        "def resolve_model_robustness_readiness(",
        "def disabled_model_robustness_readiness(", 1,
    )
    check_true("missing_multiseed_stage_owner_blocked",
               bool(check_model_stage_readiness_architecture_sources(edited)))
    edited = dict(sources)
    # Preview, post-upstream and pre-publication gates all require the owner.
    edited[RESEARCH_APPLICATION] = edited[RESEARCH_APPLICATION].replace(
        "resolve_model_robustness_readiness(",
        "hidden_robustness_plan(",
    )
    check_true("application_multiseed_owner_bypass_blocked",
               bool(check_model_stage_readiness_architecture_sources(edited)))
    edited = dict(sources)
    edited[RESEARCH_APPLICATION] = edited[RESEARCH_APPLICATION].replace(
        "        seed_states, verified_sources = _observe_model_robustness_sources(",
        "        ghost_score = object()\n"
        "        seed_states, verified_sources = _observe_model_robustness_sources(", 1,
    )
    check_true("application_multiseed_fake_source_blocked",
               bool(check_model_stage_readiness_architecture_sources(edited)))

    # Phase 3 SCORE source must remain independent of Audit. Structural
    # mutations simulate future consumer-side copies under unrelated names.
    from .research_lifecycle_architecture import (
        MODEL_SCORE_OWNER, ROLLING_MODEL_SOURCE_OWNER,
        check_model_score_owner_architecture_sources,
    )
    score_sources = dict(sources)
    for path in (MODEL_SCORE_OWNER, ROLLING_MODEL_SOURCE_OWNER):
        score_sources[path] = (root / path).read_text(encoding="utf-8")
    check("canonical_pit_score_source_without_audit_dependency", (),
          check_model_score_owner_architecture_sources(score_sources))
    for index, (path, mutation) in enumerate((
        (RESEARCH_APPLICATION,
         "\ndef shadow_pit_score_ready(manifest, score):\n"
         "    return build_file_manifest(score) == manifest['artifacts']['scores']\n"),
        (MODEL_SCORE_OWNER,
         "\ndef load_selection_point_in_time_score_source_contract_shadow():\n"
         "    return load_selection_point_in_time_ranking_contract('shadow')\n"),
    )):
        edited = dict(score_sources)
        edited[path] += mutation
        if index == 1:
            edited[MODEL_SCORE_OWNER] = edited[MODEL_SCORE_OWNER].replace(
                "def load_selection_point_in_time_score_source_contract(",
                "def deleted_pit_score_source_contract(", 1,
            )
        check_true(f"second_score_lifecycle_mutation_{index}_blocked",
                   bool(check_model_score_owner_architecture_sources(edited)))

    edited = dict(score_sources)
    edited[ROLLING_MODEL_SOURCE_OWNER] = edited[ROLLING_MODEL_SOURCE_OWNER].replace(
        "        score = load_selection_point_in_time_score_source_contract(",
        "        score = hidden_second_score_owner(", 1,
    )
    check_true("rolling_fallback_cannot_bypass_score_owner",
               bool(check_model_score_owner_architecture_sources(edited)))
    edited = dict(score_sources)
    edited[RESEARCH_APPLICATION] = edited[RESEARCH_APPLICATION].replace(
        "    resolved = load_rolling_model_source(",
        "    resolved = hidden_application_selector(", 1,
    )
    check_true("rolling_model_consumer_cannot_bypass_shared_selector",
               bool(check_model_score_owner_architecture_sources(edited)))

    # Both consumers must use one source selector without invalidating SCORE
    # when the downstream audit is missing, or reading SCORE twice when healthy.
    from services.breakout_quality import rolling_model_source as source_adapter
    synthetic_kwargs = {
        "project_root": root,
        "filter_id": "synthetic_filter",
        "model_architecture": "synthetic_architecture",
        "experiment_profile": "synthetic_profile",
        "point_in_time_dir_override": Path("synthetic_pit"),
        "expected_seed": 101,
    }
    complete_proof = SimpleNamespace(seed=101, audit={"status": "RESULT_AVAILABLE"})
    with patch.object(source_adapter, "load_selection_point_in_time_ranking_contract",
                      return_value=complete_proof), patch.object(
        source_adapter, "load_selection_point_in_time_score_source_contract",
        side_effect=AssertionError("complete Audit must not re-validate SCORE"),
    ):
        complete_source = source_adapter.load_rolling_model_source(**synthetic_kwargs)
    check_true("healthy_rolling_audit_uses_one_complete_source",
               complete_source.source_contract is complete_proof
               and complete_source.complete_contract is complete_proof
               and complete_source.audit_error is None)
    score_proof = SimpleNamespace(seed=101, manifest={"seed": 101})
    with patch.object(source_adapter, "load_selection_point_in_time_ranking_contract",
                      side_effect=FileNotFoundError("audit missing")), patch.object(
        source_adapter, "load_selection_point_in_time_score_source_contract",
        return_value=score_proof,
    ) as score_loader:
        score_only = source_adapter.load_rolling_model_source(**synthetic_kwargs)
    check_true("rolling_missing_audit_reuses_only_verified_score",
               score_only.source_contract is score_proof
               and score_only.complete_contract is None
               and "audit missing" in str(score_only.audit_error)
               and score_only.score_error is None)
    check("rolling_score_only_seed_is_from_requested_robustness_seed", 101,
          score_loader.call_args.kwargs.get("expected_seed"))
    with patch.object(source_adapter, "load_selection_point_in_time_ranking_contract",
                      side_effect=FileNotFoundError("audit missing")), patch.object(
        source_adapter, "load_selection_point_in_time_score_source_contract",
        side_effect=ValueError("score SHA invalid"),
    ):
        invalid = source_adapter.load_rolling_model_source(**synthetic_kwargs)
    check_true("invalid_rolling_score_never_infers_evidence_readiness",
               invalid.source_contract is None and invalid.complete_contract is None
               and "score SHA invalid" in str(invalid.score_error))

    # SCORE is a legitimate Research artifact with no Audit file. Complete
    # strategy Ranking still requires a source-hash-bound Audit and its gate.
    from hashlib import sha256
    import json
    from filters.breakout_quality import ranking_score_store as score_owner
    from core.breakout_quality_policy import get_breakout_quality_workflow_settings
    from core.breakout_quality_registry import get_breakout_quality_experiment_profile
    from core.breakout_quality_runtime import get_continuous_ranker_score_output_columns
    from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
    from filters.breakout_quality.paths import (
        SELECTION_POINT_IN_TIME_SCORE_FILENAME,
        SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
        SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
        SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME,
    )
    settings = get_breakout_quality_workflow_settings()
    profile = get_breakout_quality_experiment_profile(settings.experiment_profile)

    def _record(path):
        raw = path.read_bytes()
        return {"path": str(path.resolve()), "filename": path.name,
                "sha256": sha256(raw).hexdigest(), "size_bytes": len(raw)}

    with TemporaryDirectory() as td:
        directory = Path(td).resolve()
        score = directory / SELECTION_POINT_IN_TIME_SCORE_FILENAME
        coverage = directory / SELECTION_POINT_IN_TIME_COVERAGE_FILENAME
        manifest_file = directory / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
        audit_file = directory / SELECTION_POINT_IN_TIME_AUDIT_JSON_FILENAME
        score.write_text("ticker,date,breakout_quality_score\n0050,2021-01-04,0.7\n", encoding="utf-8")
        coverage.write_text("fold_id,groups\nfold_20210104_20211231,1\n", encoding="utf-8")
        target_identity = "synthetic_target"
        embedded = {"target_contract": {"schema": "fixture-v1"}}
        manifest = {
            "filter_id": settings.filter_id,
            "model_architecture": settings.model_architecture,
            "experiment_profile": settings.experiment_profile,
            "training_sample_scope": profile.training_sample_scope,
            "seed": int(settings.seed),
            "status": "BUILT",
            "continuous_target_id": target_identity,
            "source_continuous_target": embedded,
            "score_period": {"start": "2021-01-04", "end": "2021-01-04"},
            "coverage": {"scored_group_count": 1},
            "artifacts": {"scores": _record(score), "coverage": _record(coverage)},
            "score_columns": get_continuous_ranker_score_output_columns(
                profile.training_objective
            ),
            "score_eligibility_contract": build_score_eligibility_contract(profile),
        }
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        kwargs = {
            "project_root": directory,
            "filter_id": str(settings.filter_id),
            "model_architecture": str(settings.model_architecture),
            "experiment_profile": str(settings.experiment_profile),
            "point_in_time_dir_override": directory,
        }
        with patch.object(score_owner, "validate_current_pit_score_source_lineage"):
            validated_score = score_owner.load_selection_point_in_time_score_source_contract(
                **kwargs, expected_seed=int(settings.seed)
            )
            check_true("score_independent_of_audit", not audit_file.exists()
                       and validated_score.score_path == score)
            try:
                score_owner.load_selection_point_in_time_ranking_contract(
                    **kwargs, require_model_validation_pass=False
                )
                unapproved_audit_passed = True
            except FileNotFoundError:
                unapproved_audit_passed = False
            check_true("strategy_rejects_missing_audit", not unapproved_audit_passed)
            # A malformed score period may parse as NaT without raising; this
            # must never qualify as a reusable PIT SCORE source.
            broken_period = dict(manifest)
            broken_period["score_period"] = {"start": None, "end": "2021-01-04"}
            manifest_file.write_text(json.dumps(broken_period), encoding="utf-8")
            try:
                score_owner.load_selection_point_in_time_score_source_contract(**kwargs)
                accepted_missing_period = True
            except ValueError:
                accepted_missing_period = False
            check_true("missing_score_period_blocks_score_reuse", not accepted_missing_period)
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            coverage.write_text("tampered coverage", encoding="utf-8")
            try:
                score_owner.load_selection_point_in_time_score_source_contract(**kwargs)
                accepted_corrupt_coverage = True
            except ValueError:
                accepted_corrupt_coverage = False
            check_true("coverage_hash_tamper_blocks_score_reuse", not accepted_corrupt_coverage)
            coverage.write_text("fold_id,groups\nfold_20210104_20211231,1\n", encoding="utf-8")
            from services.research import breakout_quality_application as app
            from unittest.mock import MagicMock
            with patch.object(app, "PROJECT_ROOT", directory), patch.object(
                app, "_rolling_pit_dir_for_loader", return_value=directory,
            ), patch.object(
                app, "_model_robustness_dirs", return_value=(directory, directory),
            ), patch.object(
                app, "pit_fold_fitting_settings_issues", return_value=(),
            ):
                rolling_source, rolling_evidence, rolling_reason = (
                    app._load_reusable_rolling_standard_report(settings, None)
                )
                check_true("single_model_refreshes_audit_without_score_rebuild",
                           isinstance(rolling_source, score_owner.SelectionPointInTimeScoreSourceContract)
                           and rolling_evidence is None
                           and "Audit refresh required" in str(rolling_reason))
                seed_source, seed_reason = app._rolling_robustness_score_core_contract(
                    settings, seed=int(settings.seed)
                )
                check_true("seed_core_uses_same_score_owner",
                           seed_source is not None and seed_reason is None)
                missing_seed_evidence, _, _ = app._load_rolling_robustness_seed(
                    settings, seed=int(settings.seed)
                )
                check_true("seed_evidence_not_fabricated_without_audit",
                           missing_seed_evidence is None)
                calls = []
                def _capture_producer(command, _argv, **_kwargs):
                    calls.append(command)
                    return 0
                with patch.object(
                    app, "_load_forward_robustness_seed", return_value=(None, None, None),
                ), patch.object(app, "_run_command", side_effect=_capture_producer):
                    action_result = app._build_rolling_robustness_seed(
                        "synthetic", settings, seed=int(settings.seed)
                    )
                check("robustness_audit_only_does_not_rebuild_score",
                      (0, ["audit-point-in-time-scores"]),
                      (action_result, calls))
            stage = resolve_model_artifact_readiness(
                mode="rolling", source_contract=validated_score,
                evidence_payload=None, source_reason="Audit missing", source_path=None,
            )
            check("score_ready_only_requires_evaluation_refresh",
                  ("NOT_REQUIRED", "REUSE", "DERIVE"),
                  tuple(stage.plan.action_by_key[key].action
                        for key in ("FIT", "SCORE", "EVIDENCE")))
            source_artifacts = {
                "point_in_time_manifest": _record(manifest_file),
                "point_in_time_scores": _record(score),
                "point_in_time_coverage": _record(coverage),
                "continuous_target_manifest": {
                    "source": "embedded_in_point_in_time_manifest",
                    "target_id": target_identity,
                    "target_contract": embedded["target_contract"],
                },
            }
            audit = {
                "filter_id": settings.filter_id,
                "model_architecture": settings.model_architecture,
                "experiment_profile": settings.experiment_profile,
                "schema_version": 4,
                "status": "RESULT_AVAILABLE",
                "workflow": {"training_sample_scope": profile.training_sample_scope},
                "source_artifacts": source_artifacts,
                "continuous_target_id": target_identity,
                "score_period": manifest["score_period"],
                "score_group_count": 1,
                "decision_contract": {"primary_metric_scope": "pass_only_target"},
                "metrics": {"pass_only_target": {
                    "global_spearman": 0.2, "mean_daily_spearman": 0.1,
                }},
                "direction_summary": {
                    "valid_year_count": 1,
                    "positive_spearman_year_count": 1,
                    "positive_spread_year_count": 1,
                },
            }
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            complete = score_owner.load_selection_point_in_time_ranking_contract(
                **kwargs, require_model_validation_pass=True
            )
            check_true("audit_completes_ranking_without_new_score", complete.score_path == score
                       and complete.model_validation_gate["status"] == "PASS")
            # The healthy multi-seed path must not independently hash SCORE a
            # second time; the full contract already consumes the SCORE owner.
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
            with patch.object(app, "PROJECT_ROOT", directory), patch.object(
                app, "_model_robustness_dirs", return_value=(directory, directory),
            ), patch.object(
                app, "pit_fold_fitting_settings_issues", return_value=(),
            ), patch.object(
                app, "_validate_robustness_standard_payload", return_value=(),
            ), patch.object(
                app, "_rolling_robustness_score_core_contract",
                side_effect=AssertionError("SCORE-only fallback used for complete Ranking"),
            ):
                complete_seed, _, complete_reason = app._load_rolling_robustness_seed(
                    settings, seed=int(settings.seed)
                )
            check_true("healthy_multiseed_uses_single_full_score_validation",
                       complete_seed is not None and complete_reason is None)
            score.write_text("tampered", encoding="utf-8")
            try:
                score_owner.load_selection_point_in_time_score_source_contract(**kwargs)
                accepted_corrupt_score = True
            except ValueError:
                accepted_corrupt_score = False
            check_true("score_hash_tamper_blocks_score_only", not accepted_corrupt_score)
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
            try:
                score_owner.load_selection_point_in_time_ranking_contract(
                    **kwargs, require_model_validation_pass=False
                )
                accepted_corrupt_complete = True
            except ValueError:
                accepted_corrupt_complete = False
            check_true("score_hash_tamper_blocks_complete_ranking", not accepted_corrupt_complete)

    # Use completed synthetically validated evidence for the non-file-facing
    # stage action test; no model training, OOS or production source is touched.
    with patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()):
        good = {"standard_model_sop": {"schema": "synthetic_complete"}}
        with TemporaryDirectory() as td:
            source = Path(td) / "source.json"
            source.write_text("{}", encoding="utf-8")
            reuse = resolve_model_artifact_readiness(
                mode="rolling", source_contract=SimpleNamespace(audit_path=source),
                evidence_payload=good, source_reason=None, source_path=source,
            )
            check_true("valid_source_never_builds_fit_or_score",
                       reuse.plan.action_by_key["FIT"].action == "NOT_REQUIRED"
                       and reuse.plan.action_by_key["SCORE"].action == "REUSE"
                       and reuse.plan.action_by_key["EVIDENCE"].action == "REUSE"
                       and reuse.plan.action_by_key["REPORT"].action == "DERIVE")
            check_true("missing_report_only_requires_report_derive",
                       reuse.report_only_refresh and reuse.evidence_ready)
        audit = resolve_model_artifact_readiness(
            mode="rolling", source_contract=object(), evidence_payload=None,
            source_reason="outdated evidence", source_path=None,
        )
        check_true("rolling_audit_only_not_fit_or_score",
                   audit.audit_only_refresh
                   and audit.plan.action_by_key["FIT"].action == "NOT_REQUIRED"
                   and audit.plan.action_by_key["SCORE"].action == "REUSE"
                   and audit.plan.action_by_key["EVIDENCE"].action == "DERIVE")
        missing = resolve_model_artifact_readiness(
            mode="forward", source_contract=None, evidence_payload=None,
            source_reason="source absent", source_path=None,
            rolling_fit_candidate=True,
        )
        check_true("cross_mode_fit_candidate_is_not_reused_before_verification",
                   missing.display_action == "VERIFY FIT"
                   and missing.plan.action_by_key["FIT"].action == "RESUME"
                   and missing.plan.action_by_key["SCORE"].action == "BUILD")
    # Phase 7: all four Model routes must consume the same canonical FIT ->
    # SCORE -> EVIDENCE stages. Robustness alone aggregates verified seed
    # evidence into the final REPORT; a missing Audit cannot trigger refitting.
    from services.breakout_quality.model_artifact_readiness import (
        ModelRobustnessSeedSource, resolve_model_robustness_readiness,
    )
    with patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()):
        good = {"standard_model_sop": {"schema": "synthetic_complete"}}
        forward_seeds = (
            ModelRobustnessSeedSource(101, good, Path("reports/101.json"), None),
            ModelRobustnessSeedSource(202, good, Path("reports/202.json"), None),
        )
        forward = resolve_model_robustness_readiness(mode="forward", sources=forward_seeds)
        rolling = resolve_model_robustness_readiness(mode="rolling", sources=forward_seeds)
        check_true("forward_robustness_complete_seeds_reuse_fit_score_evidence",
            forward.display_action == "REUSE"
            and forward.evidence_ready
            and tuple(forward.plan.action_by_key[f"seed:{seed}:{stage}"].action
                      for seed in (101, 202) for stage in ("FIT", "SCORE", "EVIDENCE"))
            == ("REUSE", "REUSE", "REUSE") * 2)
        check_true("rolling_robustness_complete_seeds_do_not_assert_checkpoint_reuse",
            rolling.display_action == "REUSE"
            and tuple(rolling.plan.action_by_key[f"seed:{seed}:FIT"].action
                      for seed in (101, 202)) == ("NOT_REQUIRED", "NOT_REQUIRED"))
        check_true("robustness_report_has_exact_cross_seed_evidence_dependencies",
            forward.plan.action_by_key["REPORT"].action == "DERIVE"
            and forward.plan.action_by_key["REPORT"].dependencies
                == ("seed:101:EVIDENCE", "seed:202:EVIDENCE")
            and tuple(stage.seed for stage in forward.seeds) == (101, 202))
        audit_only = resolve_model_robustness_readiness(mode="rolling", sources=(
            forward_seeds[0],
            ModelRobustnessSeedSource(202, None, Path("audits/202.json"),
                                      "Audit missing", score_contract=SimpleNamespace(seed=202)),
        ))
        check_true("robustness_one_missing_audit_preserves_score_and_fit",
            audit_only.display_action == "REFRESH EVIDENCE"
            and audit_only.plan.action_by_key["seed:202:FIT"].action == "NOT_REQUIRED"
            and audit_only.plan.action_by_key["seed:202:SCORE"].action == "REUSE"
            and audit_only.plan.action_by_key["seed:202:EVIDENCE"].action == "DERIVE"
            and audit_only.plan.action_by_key["seed:101:EVIDENCE"].action == "REUSE")
        stale = resolve_model_robustness_readiness(mode="rolling", sources=(
            forward_seeds[0],
            ModelRobustnessSeedSource(202, None, Path("audits/202.json"),
                                      "SCORE hash mismatch"),
        ))
        check_true("robustness_score_invalid_cannot_be_evidence_only_refresh",
            stale.display_action == "BUILD/REFRESH"
            and stale.plan.action_by_key["seed:202:SCORE"].action == "BUILD"
            and stale.plan.action_by_key["seed:202:EVIDENCE"].action == "DERIVE")
        forward_stale = resolve_model_robustness_readiness(mode="forward", sources=(
            forward_seeds[0],
            ModelRobustnessSeedSource(202, None, Path("reports/202.json"),
                                      "fitting identity stale"),
        ))
        check_true("forward_missing_validated_source_never_claims_reuse",
            forward_stale.display_action == "BUILD/REFRESH"
            and forward_stale.plan.action_by_key["seed:202:FIT"].action == "BUILD"
            and forward_stale.plan.action_by_key["seed:202:SCORE"].action == "BUILD")
        try:
            resolve_model_robustness_readiness(mode="rolling", sources=(
                forward_seeds[0], forward_seeds[0],
            ))
        except ValueError as exc:
            duplicate_rejected = "unique" in str(exc)
        else:
            duplicate_rejected = False
        check_true("robustness_duplicate_seed_never_merges_provenance", duplicate_rejected)
    # Exercise the actual menu-owned multi-seed dispatch with isolated
    # synthetic producer ports. Do not fit, score or publish real artifacts.
    from contextlib import redirect_stdout
    from io import StringIO
    from services.research import breakout_quality_application as model_app
    workflow = SimpleNamespace(
        filter_id="synthetic", model_architecture="synthetic_arch",
        experiment_profile="synthetic_profile",
    )
    synthetic_report = {"standard_model_sop": {"schema": "synthetic_complete"}}
    def complete_seed(_settings, *, seed):
        return synthetic_report, Path(f"synthetic/{int(seed)}.json"), None
    def complete_observation(_settings, *, seed):
        payload, path, reason = complete_seed(_settings, seed=seed)
        return ModelRobustnessSeedSource(seed, payload, path, reason)
    with (
        patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()),
        patch.object(model_app, "get_breakout_quality_model_test_settings",
                     return_value=SimpleNamespace(model_profiles=(("MR-SYN", "synthetic_profile"),))),
        patch.object(model_app, "get_breakout_quality_workflow_settings", return_value=workflow),
        patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=(101, 202)),
        patch.object(model_app, "_load_forward_robustness_seed", side_effect=complete_seed),
        patch.object(model_app, "_inspect_rolling_robustness_seed", side_effect=complete_observation),
        patch.object(model_app, "_rolling_robustness_score_core_contract",
                     side_effect=AssertionError("complete seed must not rescore")) as no_rescore,
        patch.object(model_app, "invalidate_rolling_model_source_cache") as cache_reset,
        patch.object(model_app, "_prepare_continuous_research_inputs",
                     side_effect=AssertionError("complete seeds must not prepare")) as no_prepare,
        patch.object(model_app, "_build_forward_robustness_seed",
                     side_effect=AssertionError("complete seed must not fit")) as no_forward_build,
        patch.object(model_app, "_build_rolling_robustness_seed",
                     side_effect=AssertionError("complete seed must not build scores")) as no_rolling_build,
        patch.object(model_app, "aggregate_standard_model_sop_robustness", return_value={"schema": "synthetic"}),
        patch.object(model_app, "_render_standard_model_comparison", return_value="SYNTHETIC"),
        patch.object(model_app, "_write_standard_model_comparison_report",
                     return_value=(Path("synthetic.json"), Path("synthetic.md"))) as write_report,
        patch.object(model_app, "render_status_paths", return_value=""),
        patch.object(model_app, "_print_model_action_status") as print_status,
        redirect_stdout(StringIO()),
    ):
        forward_code = model_app._run_configured_model_robustness("synthetic", rolling=False)
        forward_action = print_status.call_args.args[0][0][2]
        rolling_code = model_app._run_configured_model_robustness("synthetic", rolling=True)
        rolling_action = print_status.call_args.args[0][0][2]
    check_true("complete_forward_rolling_robustness_dispatch_reuses_only_and_renders_once_per_mode",
        forward_code == rolling_code == 0
        and forward_action == rolling_action == "REUSE"
        and write_report.call_count == 2 and no_rescore.call_count == 0
        and cache_reset.call_count == 1  # Fresh Rolling preview, no Producer
        and no_prepare.call_count == no_forward_build.call_count == no_rolling_build.call_count == 0)

    # When exactly one Rolling seed loses Audit while retaining validated SCORE,
    # dispatch the canonical Audit refresh for that seed alone.
    calls = {202: 0}
    def interrupted_seed(_settings, *, seed):
        if seed == 202:
            calls[202] += 1
            if calls[202] == 1:
                return ModelRobustnessSeedSource(
                    seed, None, Path("synthetic/202.json"), "Audit missing",
                    score_contract=SimpleNamespace(seed=seed),
                )
        return ModelRobustnessSeedSource(
            seed, synthetic_report, Path(f"synthetic/{seed}.json"), None,
        )
    with (
        patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()),
        patch.object(model_app, "get_breakout_quality_model_test_settings",
                     return_value=SimpleNamespace(model_profiles=(("MR-SYN", "synthetic_profile"),))),
        patch.object(model_app, "get_breakout_quality_workflow_settings", return_value=workflow),
        patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=(101, 202)),
        patch.object(model_app, "_inspect_rolling_robustness_seed", side_effect=interrupted_seed),
        patch.object(model_app, "_rolling_robustness_score_core_contract",
                     side_effect=AssertionError("preview must consume validated SCORE")) as score_only,
        patch.object(model_app, "invalidate_rolling_model_source_cache") as audit_cache_reset,
        patch.object(model_app, "_prepare_continuous_research_inputs", return_value=0) as prep,
        patch.object(model_app, "_build_rolling_robustness_seed", return_value=0) as audit_build,
        patch.object(model_app, "_build_forward_robustness_seed",
                     side_effect=AssertionError("never build Forward in Rolling")) as forward_build,
        patch.object(model_app, "aggregate_standard_model_sop_robustness", return_value={"schema": "synthetic"}),
        patch.object(model_app, "_render_standard_model_comparison", return_value="SYNTHETIC"),
        patch.object(model_app, "_write_standard_model_comparison_report",
                     return_value=(Path("synthetic.json"), Path("synthetic.md"))) as final_report,
        patch.object(model_app, "render_status_paths", return_value=""),
        patch.object(model_app, "_print_model_action_status") as print_status,
        redirect_stdout(StringIO()),
    ):
        code = model_app._run_configured_model_robustness("synthetic", rolling=True)
        status_action = print_status.call_args.args[0][0][2]
    check_true("missing_one_seed_audit_refreshes_only_that_seed_then_aggregates",
        code == 0 and status_action == "REFRESH EVIDENCE"
        and score_only.call_count == 0 and prep.call_count == 0
        and audit_cache_reset.call_count == 2  # Initial preview + subprocess Audit
        and audit_build.call_count == 1 and audit_build.call_args.kwargs["seed"] == 202
        and audit_build.call_args.kwargs["inputs_prepared"] is False
        and calls[202] == 3 and forward_build.call_count == 0
        and final_report.call_count == 1)

    # Producer failure must stop before the cross-seed REPORT can be emitted.
    with (
        patch.object(model_app, "get_breakout_quality_model_test_settings",
                     return_value=SimpleNamespace(model_profiles=(("MR-SYN", "synthetic_profile"),))),
        patch.object(model_app, "get_breakout_quality_workflow_settings", return_value=workflow),
        patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=(101, 202)),
        patch.object(model_app, "_inspect_rolling_robustness_seed",
                     side_effect=lambda _settings, *, seed: ModelRobustnessSeedSource(
                         seed, None, Path("synthetic/audit.json"), "SCORE hash mismatch")),
        patch.object(model_app, "_rolling_robustness_score_core_contract",
                     side_effect=AssertionError("preview must not recheck SCORE")),
        patch.object(model_app, "_prepare_continuous_research_inputs", return_value=0),
        patch.object(model_app, "_build_rolling_robustness_seed", return_value=71) as failed_producer,
        patch.object(model_app, "aggregate_standard_model_sop_robustness",
                     side_effect=AssertionError("failed seed cannot aggregate")) as no_aggregate,
        patch.object(model_app, "_write_standard_model_comparison_report",
                     side_effect=AssertionError("failed seed cannot publish")) as no_report,
        patch.object(model_app, "_print_model_action_status"),
        redirect_stdout(StringIO()),
    ):
        failed_code = model_app._run_configured_model_robustness("synthetic", rolling=True)
    check_true("failed_seed_score_build_stops_aggregate_report_and_next_seed",
        failed_code == 71 and failed_producer.call_count == 1
        and failed_producer.call_args.kwargs["seed"] == 101
        and no_aggregate.call_count == no_report.call_count == 0)

    # Phase 8: test the actual producer routing, not only the preview plan.
    # No real Dataset/Target, FIT, SCORE or Audit artifacts are modified.
    producer_settings = SimpleNamespace(
        filter_id="synthetic", model_architecture="synthetic_arch",
        experiment_profile="synthetic_profile", point_in_time_resume=True,
        point_in_time_inner_validation_months=6,
        point_in_time_train_window_months=None,
    )
    producer_mode = SimpleNamespace(
        score_start_date="2021-01-01", fold_months=12,
        fold_anchor_date=None, single_score_block=False, score_end_date=None,
    )
    with TemporaryDirectory() as td:
        test_dir = Path(td)
        def seed_dirs(_settings, *, seed, rolling):
            directory = test_dir / ("rolling" if rolling else "forward") / str(seed)
            return directory, directory

        calls = []
        def record_producer(command, argv, **_kwargs):
            calls.append((command, tuple(argv)))
            return 0

        with (
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=producer_mode),
            patch.object(model_app, "_model_robustness_dirs", side_effect=seed_dirs),
            patch.object(model_app, "_rolling_robustness_score_core_ready", return_value=(True, None)) as score_probe,
            patch.object(model_app, "_load_forward_robustness_seed",
                         side_effect=AssertionError("audit-only must not discover Forward")) as forward_probe,
            patch.object(model_app, "_prepare_continuous_research_inputs",
                         side_effect=AssertionError("audit-only must not prepare inputs")) as prep,
            patch.object(model_app, "_run_command", side_effect=record_producer),
        ):
            audit_code = model_app._build_rolling_robustness_seed(
                "synthetic", producer_settings, seed=101,
            )
        check_true("producer_valid_score_executes_audit_only_without_upstream_or_forward",
            audit_code == 0 and [c[0] for c in calls] == ["audit-point-in-time-scores"]
            and score_probe.call_count == 1 and prep.call_count == 0 and forward_probe.call_count == 0)

        calls.clear()
        with (
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=producer_mode),
            patch.object(model_app, "_model_robustness_dirs", side_effect=seed_dirs),
            patch.object(model_app, "_rolling_robustness_score_core_ready", return_value=(False, "SHA changed")) as score_probe,
            patch.object(model_app, "_prepare_continuous_research_inputs", return_value=0) as prep,
            patch.object(model_app, "_load_forward_robustness_seed", return_value=(None, None, "absent")) as forward_probe,
            patch.object(model_app, "_append_pit_forward_checkpoint_import_args") as import_proof,
            patch.object(model_app, "_run_command", side_effect=record_producer),
        ):
            stale_code = model_app._build_rolling_robustness_seed(
                "synthetic", producer_settings, seed=101, inputs_prepared=False,
            )
        check_true("producer_score_becomes_stale_after_preview_prepares_before_score_and_audit",
            stale_code == 0 and [c[0] for c in calls] == [
                "build-point-in-time-scores", "audit-point-in-time-scores"]
            and score_probe.call_count == 1 and prep.call_count == 1
            and forward_probe.call_count == 1 and import_proof.call_count == 1
            and "--seed" in calls[0][1] and "101" in calls[0][1])

        calls.clear()
        with (
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=producer_mode),
            patch.object(model_app, "_model_robustness_dirs", side_effect=seed_dirs),
            patch.object(model_app, "_rolling_robustness_score_core_ready", return_value=(False, "stale")),
            patch.object(model_app, "_prepare_continuous_research_inputs",
                         side_effect=AssertionError("planned SCORE build already prepared")) as redundant_prep,
            patch.object(model_app, "_load_forward_robustness_seed", return_value=(None, None, None)),
            patch.object(model_app, "_append_pit_forward_checkpoint_import_args"),
            patch.object(model_app, "_run_command", side_effect=record_producer),
        ):
            prepared_code = model_app._build_rolling_robustness_seed(
                "synthetic", producer_settings, seed=202, inputs_prepared=True,
            )
        check_true("producer_prepared_score_build_does_not_repeat_upstream",
            prepared_code == 0 and redundant_prep.call_count == 0
            and [c[0] for c in calls] == ["build-point-in-time-scores", "audit-point-in-time-scores"])

        calls.clear()
        with (
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=producer_mode),
            patch.object(model_app, "_model_robustness_dirs", side_effect=seed_dirs),
            patch.object(model_app, "_rolling_robustness_score_core_ready", return_value=(False, "stale")),
            patch.object(model_app, "_prepare_continuous_research_inputs", return_value=73) as upstream_fail,
            patch.object(model_app, "_load_forward_robustness_seed",
                         side_effect=AssertionError("blocked should not probe Forward")) as forbidden_probe,
            patch.object(model_app, "_run_command", side_effect=record_producer),
        ):
            blocked_code = model_app._build_rolling_robustness_seed(
                "synthetic", producer_settings, seed=202,
            )
        check_true("producer_upstream_gate_failure_stops_before_checkpoint_and_score",
            blocked_code == 73 and upstream_fail.call_count == 1
            and forbidden_probe.call_count == 0 and not calls)

        calls.clear()
        with (
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=producer_mode),
            patch.object(model_app, "_model_robustness_dirs", side_effect=seed_dirs),
            patch.object(model_app, "_rolling_robustness_score_core_ready",
                         side_effect=AssertionError("Force Rebuild must not REUSE SCORE")) as no_reuse,
            patch.object(model_app, "_prepare_continuous_research_inputs",
                         side_effect=AssertionError("already prepared")) as no_extra_prep,
            patch.object(model_app, "_load_forward_robustness_seed",
                         side_effect=AssertionError("Force Rebuild must not import old FIT")) as no_forward,
            patch.object(model_app, "_run_command", side_effect=record_producer),
        ):
            forced_code = model_app._build_rolling_robustness_seed(
                "synthetic", producer_settings, seed=101, force_rebuild=True,
                inputs_prepared=True,
            )
        check_true("producer_force_rebuild_stays_isolated_and_rebuilds_score_audit",
            forced_code == 0 and [c[0] for c in calls] == [
                "build-point-in-time-scores", "audit-point-in-time-scores"]
            and no_reuse.call_count == no_extra_prep.call_count == no_forward.call_count == 0
            and "--no-resume" in calls[0][1]
            and calls[0][1].count("--checkpoint-cache-root") == 1)

    # Phase 9: one Rolling inspection must retain the verified SCORE returned
    # by the common source owner; never re-read SCORE solely for a preview.
    with TemporaryDirectory() as td:
        fake_score = SimpleNamespace(seed=101, manifest={"synthetic": True})
        missing_audit = SimpleNamespace(
            source_contract=fake_score, complete_contract=None,
            audit_error="Audit missing", score_error=None,
        )
        with (
            patch.object(model_app, "_model_robustness_dirs", return_value=(Path(td), Path(td))),
            patch.object(model_app, "load_rolling_model_source", return_value=missing_audit) as canonical_load,
            patch.object(model_app, "pit_fold_fitting_settings_issues", return_value=()) as fit_check,
        ):
            missing_observation = model_app._inspect_rolling_robustness_seed(workflow, seed=101)
        check_true("seed_missing_audit_uses_exactly_one_canonical_score_observation",
            canonical_load.call_count == 1 and fit_check.call_count == 1
            and missing_observation.verified_evidence is None
            and missing_observation.score_contract is fake_score
            and "EVIDENCE refresh" in str(missing_observation.reason))
        missing_stage = resolve_model_robustness_readiness(
            mode="rolling", sources=(missing_observation,),
        )
        check_true("single_rolling_observation_authorizes_evidence_only_not_fit",
            missing_stage.plan.action_by_key["seed:101:FIT"].action == "NOT_REQUIRED"
            and missing_stage.plan.action_by_key["seed:101:SCORE"].action == "REUSE"
            and missing_stage.plan.action_by_key["seed:101:EVIDENCE"].action == "DERIVE")
        with (
            patch.object(model_app, "_model_robustness_dirs", return_value=(Path(td), Path(td))),
            patch.object(model_app, "load_rolling_model_source", return_value=missing_audit),
            patch.object(model_app, "pit_fold_fitting_settings_issues", return_value=("FIT settings stale",)),
        ):
            invalid_fit = model_app._inspect_rolling_robustness_seed(workflow, seed=101)
        check_true("seed_fit_recipe_drift_cannot_reuse_score_or_evidence",
            invalid_fit.score_contract is None and invalid_fit.verified_evidence is None
            and "settings stale" in str(invalid_fit.reason))
        wrong_seed = SimpleNamespace(
            source_contract=fake_score,
            complete_contract=SimpleNamespace(seed=202, audit=synthetic_report),
            audit_error=None, score_error=None,
        )
        with (
            patch.object(model_app, "_model_robustness_dirs", return_value=(Path(td), Path(td))),
            patch.object(model_app, "load_rolling_model_source", return_value=wrong_seed),
            patch.object(model_app, "pit_fold_fitting_settings_issues",
                         side_effect=AssertionError("seed mismatch must fail before FIT settings")) as no_fit,
        ):
            wrong = model_app._inspect_rolling_robustness_seed(workflow, seed=101)
        check_true("complete_seed_mismatch_does_not_turn_into_score_only_reuse",
            wrong.score_contract is None and wrong.verified_evidence is None
            and no_fit.call_count == 0)
        complete_audit = SimpleNamespace(
            source_contract=fake_score,
            complete_contract=SimpleNamespace(seed=101, audit=synthetic_report),
            audit_error=None, score_error=None,
        )
        with (
            patch.object(model_app, "_model_robustness_dirs", return_value=(Path(td), Path(td))),
            patch.object(model_app, "load_rolling_model_source", return_value=complete_audit) as canonical_load,
            patch.object(model_app, "pit_fold_fitting_settings_issues", return_value=()),
            patch.object(model_app, "_validate_robustness_standard_payload", return_value=()),
        ):
            complete_observation = model_app._inspect_rolling_robustness_seed(workflow, seed=101)
        check_true("complete_seed_reuses_full_audit_without_secondary_score_validation",
            canonical_load.call_count == 1
            and complete_observation.verified_evidence == synthetic_report
            and complete_observation.score_contract is None)

    # Source-owner API delegates invalidation to the cached canonical Ranking
    # contract; a subprocess cannot invalidate the parent process implicitly.
    from services.breakout_quality import rolling_model_source as rolling_owner
    with patch.object(
        rolling_owner, "clear_selection_point_in_time_ranking_contract_cache",
    ) as owner_clear:
        rolling_owner.invalidate_rolling_model_source_cache()
    check_true("rolling_source_cache_invalidation_uses_canonical_cache_owner",
               owner_clear.call_count == 1)

    with patch.object(
        model_app, "_observe_model_robustness_seed",
        side_effect=lambda _settings, *, seed, rolling: ModelRobustnessSeedSource(
            seed + 1, synthetic_report, Path(f"synthetic/{seed + 1}.json"), None,
        ),
    ):
        try:
            model_app._observe_model_robustness_sources(
                workflow, seeds=(101, 202), rolling=True,
            )
            wrong_seed_order_accepted = True
        except RuntimeError:
            wrong_seed_order_accepted = False
    check_true("verified_seed_observation_must_match_configured_order_and_identity",
               not wrong_seed_order_accepted)

    # Dataset/Target preparation and per-seed Producers are mutation barriers.
    # Both must force new canonical observations before stale evidence is
    # consumed, without rereading fully READY runs.
    def run_phase9_synthetic_routing(*, initially_ready, after_prepare_ready,
                                     after_first_build_stale, final_stale_first=False):
        phase = {"prepared": False, "built": [], "observed": []}
        def observe(_settings, *, seed, rolling):
            phase["observed"].append((int(seed), phase["prepared"], tuple(phase["built"])))
            ready = initially_ready[int(seed)] if not phase["prepared"] else after_prepare_ready[int(seed)]
            if int(seed) == 202 and phase["built"] and after_first_build_stale:
                ready = False
            if int(seed) in phase["built"]:
                ready = True
            if int(seed) == 101 and 202 in phase["built"] and final_stale_first:
                ready = False  # last seed's Producer invalidated an earlier seed
            return ModelRobustnessSeedSource(
                int(seed), synthetic_report if ready else None,
                Path(f"synthetic/{int(seed)}.json"), None if ready else "lineage stale",
            )
        def prepare(_name, _settings):
            phase["prepared"] = True
            return 0
        def build(_name, _settings, *, seed, **_kwargs):
            phase["built"].append(int(seed))
            return 0
        with (
            patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()),
            patch.object(model_app, "get_breakout_quality_model_test_settings",
                         return_value=SimpleNamespace(model_profiles=(("MR-SYN", "synthetic_profile"),))),
            patch.object(model_app, "get_breakout_quality_workflow_settings", return_value=workflow),
            patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=(101, 202)),
            patch.object(model_app, "_observe_model_robustness_seed", side_effect=observe),
            patch.object(model_app, "_prepare_continuous_research_inputs", side_effect=prepare) as prep,
            patch.object(model_app, "_build_forward_robustness_seed", side_effect=build) as producer,
            patch.object(model_app, "_load_forward_robustness_seed",
                         side_effect=lambda _settings, *, seed: (synthetic_report, Path(f"synthetic/{seed}.json"), None)),
            patch.object(model_app, "aggregate_standard_model_sop_robustness",
                         return_value={"schema": "synthetic"}) as aggregate,
            patch.object(model_app, "_render_standard_model_comparison", return_value="SYNTHETIC"),
            patch.object(model_app, "_write_standard_model_comparison_report",
                         return_value=(Path("synthetic.json"), Path("synthetic.md"))) as published,
            patch.object(model_app, "render_status_paths", return_value=""),
            patch.object(model_app, "_print_model_action_status"),
            redirect_stdout(StringIO()),
        ):
            try:
                status = model_app._run_configured_model_robustness("synthetic", rolling=False)
                error = None
            except RuntimeError as exc:
                status, error = None, str(exc)
        return (status, phase, prep.call_count, producer.call_count,
                published.call_count, aggregate.call_count, error)

    status, phase, prep_count, build_count, report_count, agg_count, error = run_phase9_synthetic_routing(
        initially_ready={101: True, 202: True},
        after_prepare_ready={101: True, 202: True}, after_first_build_stale=False,
    )
    check_true("all_ready_seed_sources_checked_once_without_upstream_or_rebuild",
        status == 0 and prep_count == build_count == 0 and report_count == 1
        and phase["observed"] == [(101, False, ()), (202, False, ())])
    status, phase, prep_count, build_count, report_count, agg_count, error = run_phase9_synthetic_routing(
        initially_ready={101: True, 202: False},
        after_prepare_ready={101: False, 202: False}, after_first_build_stale=False,
    )
    check_true("dataset_preparation_replans_previously_ready_seed_before_report",
        status == 0 and prep_count == 1 and build_count == 2 and report_count == 1
        and phase["built"] == [101, 202]
        and (101, True, ()) in phase["observed"])
    status, phase, prep_count, build_count, report_count, agg_count, error = run_phase9_synthetic_routing(
        initially_ready={101: False, 202: True},
        after_prepare_ready={101: False, 202: True}, after_first_build_stale=True,
    )
    check_true("earlier_seed_producer_invalidates_later_seed_reuse_before_aggregate",
        status == 0 and prep_count == 1 and build_count == 2 and report_count == agg_count == 1
        and error is None and phase["built"] == [101, 202]
        and (202, True, (101,)) in phase["observed"])
    status, phase, prep_count, build_count, report_count, agg_count, error = run_phase9_synthetic_routing(
        initially_ready={101: False, 202: True},
        after_prepare_ready={101: False, 202: True},
        after_first_build_stale=True, final_stale_first=True,
    )
    check_true("last_seed_mutation_of_prior_source_blocks_report_publication",
        status is None and prep_count == 1 and build_count == 2
        and agg_count == report_count == 0
        and "拒絕發布報表" in str(error) and "seed=101" in str(error))

    # Multi-model preview happens before any per-model Producer. A model built
    # earlier may invalidate a later model's precomputed stage plan, and a model
    # built later can invalidate an earlier model's already aggregated evidence.
    def run_cross_model_revalidation(*, corrupt_prior_after_later_build):
        phase = {"built": [], "observed": []}
        profiles = (("MR-SYN-A", "phase9_a"), ("MR-SYN-B", "phase9_b"))
        def profile_settings(*, experiment_profile):
            return SimpleNamespace(
                filter_id="synthetic", model_architecture="synthetic_arch",
                experiment_profile=experiment_profile,
            )
        def observed(settings, *, seed, rolling):
            profile = str(settings.experiment_profile)
            phase["observed"].append((profile, int(seed), tuple(phase["built"])))
            if profile == "phase9_a":
                ready = seed == 202 or (profile, seed) in phase["built"]
                if seed == 202 and ("phase9_b", 101) in phase["built"] and corrupt_prior_after_later_build:
                    ready = False
            else:
                ready = seed == 202 or ("phase9_a", 101) not in phase["built"] or (profile, seed) in phase["built"]
            return ModelRobustnessSeedSource(
                int(seed), synthetic_report if ready else None,
                Path(f"synthetic/{profile}/{seed}.json"), None if ready else "lineage stale",
            )
        def producer(_program, settings, *, seed, **_kwargs):
            phase["built"].append((str(settings.experiment_profile), int(seed)))
            return 0
        with (
            patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()),
            patch.object(model_app, "get_breakout_quality_model_test_settings",
                         return_value=SimpleNamespace(model_profiles=profiles)),
            patch.object(model_app, "get_breakout_quality_workflow_settings", side_effect=profile_settings),
            patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=(101, 202)),
            patch.object(model_app, "_observe_model_robustness_seed", side_effect=observed),
            patch.object(model_app, "_prepare_continuous_research_inputs", return_value=0) as prepare,
            patch.object(model_app, "_build_forward_robustness_seed", side_effect=producer),
            patch.object(model_app, "_load_forward_robustness_seed", side_effect=lambda settings, *, seed: (
                synthetic_report, Path(f"synthetic/{settings.experiment_profile}/{seed}.json"), None,
            )),
            patch.object(model_app, "aggregate_standard_model_sop_robustness",
                         return_value={"schema": "synthetic"}) as aggregate,
            patch.object(model_app, "_render_standard_model_comparison", return_value="SYNTHETIC"),
            patch.object(model_app, "_write_standard_model_comparison_report",
                         return_value=(Path("synthetic.json"), Path("synthetic.md"))) as publish,
            patch.object(model_app, "render_status_paths", return_value=""),
            patch.object(model_app, "_print_model_action_status"),
            redirect_stdout(StringIO()),
        ):
            try:
                code = model_app._run_configured_model_robustness("synthetic", rolling=False)
                error = None
            except RuntimeError as exc:
                code, error = None, str(exc)
        return code, error, phase, prepare.call_count, aggregate.call_count, publish.call_count

    code, error, phase, prepare_count, aggregate_count, publish_count = run_cross_model_revalidation(
        corrupt_prior_after_later_build=False,
    )
    check_true("later_model_reobserves_plan_after_prior_model_producer",
        code == 0 and error is None
        and phase["built"] == [("phase9_a", 101), ("phase9_b", 101)]
        and prepare_count == 2 and aggregate_count == 3 and publish_count == 1
        and ("phase9_b", 101, ()) in phase["observed"]
        and ("phase9_b", 101, (("phase9_a", 101),)) in phase["observed"])
    code, error, phase, prepare_count, aggregate_count, publish_count = run_cross_model_revalidation(
        corrupt_prior_after_later_build=True,
    )
    check_true("later_model_producer_cannot_publish_stale_earlier_model_evidence",
        code is None and prepare_count == 2 and aggregate_count == 2
        and publish_count == 0
        and "MR-SYN-A" in str(error) and "拒絕發布報表" in str(error))

    # Normal Single resolver does not infer source readiness from unattested
    # payload, even though the Robustness adapter can pass validated evidence.
    bare = resolve_model_artifact_readiness(
        mode="forward", source_contract=None,
        evidence_payload={"standard_model_sop": {"schema": "synthetic_complete"}},
        source_reason="no canonical source", source_path=None,
    )
    check_true("single_missing_source_cannot_infer_reuse_from_payload",
               not bare.source_ready and not bare.evidence_ready)

    # Report publication is the last, independent dependency.  Tampering or
    # removing the renderer output must not invalidate FIT/SCORE/EVIDENCE.
    from services.breakout_quality.standard_model_sop import standard_model_sop_completeness_issues
    from services.breakout_quality.model_artifact_readiness import standard_model_report_matches_source
    from services.research.breakout_quality_application import _write_standard_model_sop_report
    import json
    scopes = ("validation", "oos", "breakout_candidate_oos")
    m1 = {
        "group_count": 8, "mean_daily_spearman": 0.3,
        "global_spearman_vs_raw_target": 0.2,
        "pairwise_concordance": 0.58,
        "top_score_decile_raw_target_mean": 1.5,
        "bottom_score_decile_raw_target_mean": 0.5,
    }
    m3m4 = {
        "target_to_full_mfe_daily_spearman": 0.5,
        "target_to_low_adverse_daily_spearman": 0.3,
        "score_to_full_mfe_daily_spearman": 0.4,
        "score_to_low_adverse_daily_spearman": 0.1,
        "joint_quality_evaluation_id": "same_day_min_mfe_safety_percentile_v1",
        "top_10pct": {
            "n": 8, "mfe_pct_mean": 20.0, "adverse_pct_mean": 4.0,
            "joint_quality_pct": 70.0,
        },
    }
    standard = {
        "schema": "standard_model_sop_v10", "evaluation_mode": "rolling_oos",
        "training": {"objective": "synthetic"},
        "split_metrics": {key: m1 for key in scopes},
        "upside_downside_alignment_evaluation": {key: m3m4 for key in scopes},
        "evaluation_truth": {
            "horizon_bars": 40, "mfe_source": "target_favorable_return",
            "adverse_source": "target_adverse_return_to_peak",
            "joint_quality_id": "same_day_min_mfe_safety_percentile_v1",
            "cohort_identity": {key: "f" * 64 for key in scopes},
        },
    }
    witness = {"model_research_id": "SYNTHETIC", "standard_model_sop": standard}
    check("independent_standard_evidence_ready", (),
          standard_model_sop_completeness_issues(witness))
    settings = SimpleNamespace(
        filter_id="breakout_quality_v1", model_architecture="inception_time_v1",
        experiment_profile="synthetic",
    )
    identity = {
        "filter_id": settings.filter_id, "architecture": settings.model_architecture,
        "profile": settings.experiment_profile,
    }
    with TemporaryDirectory() as td:
        source = Path(td) / "audit.json"
        source.write_text("{}", encoding="utf-8")
        first = resolve_model_artifact_readiness(
            mode="rolling", source_contract=object(), evidence_payload=witness,
            source_reason=None, source_path=source, report_identity=identity,
        )
        check("missing_renderer_only_derives_report",
              ("NOT_REQUIRED", "REUSE", "REUSE", "DERIVE"),
              tuple(first.plan.action_by_key[key].action
                    for key in ("FIT", "SCORE", "EVIDENCE", "REPORT")))
        files = _write_standard_model_sop_report(
            settings, source_payload=witness, source_path=source,
        )
        mtime = tuple(file.stat().st_mtime_ns for file in files)
        second = resolve_model_artifact_readiness(
            mode="rolling", source_contract=object(), evidence_payload=witness,
            source_reason=None, source_path=source, report_identity=identity,
        )
        check_true("approved_renderer_artifacts_are_reusable",
                   second.plan.action_by_key["REPORT"].action == "REUSE")
        _write_standard_model_sop_report(settings, source_payload=witness, source_path=source)
        check("report_reuse_does_not_rewrite_history", mtime,
              tuple(file.stat().st_mtime_ns for file in files))
        tampered = json.loads(files[0].read_text(encoding="utf-8"))
        tampered["profile"] = "different-profile"
        files[0].write_text(json.dumps(tampered), encoding="utf-8")
        check_true("report_identity_drift_detected",
                   not standard_model_report_matches_source(
                       witness, source, expected_identity=identity,
                   ))
        after_tamper = resolve_model_artifact_readiness(
            mode="forward", source_contract=object(), evidence_payload=witness,
            source_reason=None, source_path=source, report_identity=identity,
        )
        check("tampering_only_refreshes_report_not_fit_or_score",
              ("REUSE", "REUSE", "REUSE", "DERIVE"),
              tuple(after_tamper.plan.action_by_key[key].action
                    for key in ("FIT", "SCORE", "EVIDENCE", "REPORT")))
    return results, summary
