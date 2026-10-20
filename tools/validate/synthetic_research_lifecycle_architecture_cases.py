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
    # AI: Test the shared provenance seam, not fixed experiment identities.
    for module_name in (FIT_OWNER, PIT_PRODUCER):
        disconnected = dict(sources)
        disconnected[module_name] = disconnected[module_name].replace(
            "from services.breakout_quality.fitting_data_dependencies import",
            "from services.breakout_quality.unregistered_dependency import", 1,
        )
        check_true("fit_dependency_ssot_import_required_" + module_name.rsplit("/", 1)[-1],
                   bool(check_research_fit_architecture_sources(disconnected)))
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

    # AI: A single dependency snapshot must project to the *unchanged* legacy
    # Forward and native PIT Fold source schemas for every registered profile.
    # This is an identity-preserving engineering refactor, not a migration.
    from services.breakout_quality.fitting_data_dependencies import FitDataDependencies
    from filters.breakout_quality.model_source_provenance import canonical_source_dataset_artifacts
    from core.breakout_quality_registry import SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
    projection_issues = []
    for profile_name in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES:
        summary = {
            "dataset": "synthetic", "policy": {"id": "p"},
            "training_universe_start_date": "2017-01-03",
            "source_data_inventory": {"csv_inventory_sha256": "v"},
            "dataset_storage_schema_version": 2,
            "dataset_artifacts": {"events": "a"},
            "source_dataset_artifacts": {"events": "b"},
            "input_price_source": {"source": "raw"},
            "input_trade_count_source": {"source": "pit"},
        }
        target = {
            "artifacts": None, "schema_version": 3, "target_id": "synthetic",
            "target_contract": {"target": "T"}, "context_features": ["A"],
            "pair_weight_context_features": ["B"],
            "pair_weight_context_contract": {"kind": "C"},
            "risk_param_coverage_start": "2016-01-01",
            "predicted_upside_context_manifest": {"artifact": "U"},
            "predicted_safety_context_manifest": {"artifact": "S"},
        }
        fixture = SimpleNamespace(summary=summary, target_manifest=target)
        old_forward = {
            "dataset": summary.get("dataset"), "dataset_policy": summary.get("policy"),
            "training_universe_start_date": summary.get("training_universe_start_date"),
            "source_data_inventory": summary.get("source_data_inventory"),
            "input_price_source": summary["input_price_source"],
            "input_trade_count_source": summary["input_trade_count_source"],
            "source_dataset_artifacts": canonical_source_dataset_artifacts(summary),
            "source_target_artifacts": target.get("artifacts"),
            "target_id": target.get("target_id"), "target_contract": target.get("target_contract"),
            "context_features": target.get("context_features"),
            "pair_weight_context_features": target.get("pair_weight_context_features"),
            "pair_weight_context_contract": target.get("pair_weight_context_contract"),
            "risk_param_coverage_start": target.get("risk_param_coverage_start"),
            "predicted_upside_context_manifest": target.get("predicted_upside_context_manifest"),
            "predicted_safety_context_manifest": target.get("predicted_safety_context_manifest"),
        }
        old_fold = {
            "dataset_policy": summary.get("policy"),
            "dataset_storage_schema_version": summary.get("dataset_storage_schema_version"),
            "source_data_inventory": summary.get("source_data_inventory"),
            "input_price_source": summary["input_price_source"],
            "input_trade_count_source": summary["input_trade_count_source"],
            "dataset_artifacts": summary["source_dataset_artifacts"],
            "target_schema_version": target.get("schema_version"),
            "target_contract": target.get("target_contract"),
            "target_artifacts": target.get("artifacts"),
            "training_universe_start_date": "2017-01-03",
        }
        snapshot = FitDataDependencies.from_bundle(fixture)
        if (owner.source_training_identity(fixture) != old_forward
            or snapshot.forward_source() != old_forward
            or snapshot.rolling_source(daily_training_universe_start_date="2017-01-03") != old_fold
            or "training_universe_start_date" in snapshot.rolling_source()):
            projection_issues.append(profile_name)
    check("fit_data_dependency_ssot_preserves_all_registry_profile_source_bytes", (),
          tuple(projection_issues))

    # AI: Reverse Rolling -> Forward FIT preserves the historical Fold FitKey.
    # Event Dataset hash changes only pass with fully proven FIT inputs.
    historical = deepcopy(identity)
    # AI: Native checkpoint fixtures use a registered capability profile. A
    # dummy profile cannot prove which optional context is consumed by fitting.
    from core.breakout_quality_registry import (
        SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
        TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
    )
    from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
    from core.breakout_quality_runtime import (
        CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
        CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
    )
    no_context_profile = next(
        profile_name for profile_name in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
        if get_breakout_quality_experiment_profile(profile_name).training_sample_scope
           == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        and get_continuous_ranker_execution_recipe(profile_name).context_policy.source
           == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
        and get_continuous_ranker_execution_recipe(profile_name).target_policy.materialization_mode
           == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT
    )
    historical["experiment_profile"] = no_context_profile
    historical["source_contract"] = {
        "dataset_policy": "canonical", "dataset_artifacts": {"feature_bank": "sha-old"},
        "target_artifacts": {"raw_target": "sha-target"},
        "source_data_inventory": {"csv_inventory_sha256": "legacy"},
    }
    current = deepcopy(historical)
    current["source_contract"]["source_data_inventory"] = {
        "csv_inventory_sha256": "current", "ohlcv_sanitization_contract_version": 2,
    }
    check_true("reverse_bridge_without_root_rejects_unproven_inventory",
               "source_contract.source_data_inventory mismatch" in
               owner._first_fold_fit_identity_issues(historical, current))
    with TemporaryDirectory() as td:
        root_dir = Path(td)
        with patch.object(owner, "source_inventory_matches_current", return_value=True) as proof:
            check("reverse_bridge_verified_inventory_same_fitting_identity", (),
                  owner._first_fold_fit_identity_issues(
                      historical, current, project_root=root_dir,
                  ))
            check_true("reverse_bridge_delegates_to_canonical_source_proof",
                       proof.call_count == 1 and
                       proof.call_args.kwargs["project_root"] == root_dir and
                       proof.call_args.kwargs["published"] == historical["source_contract"]["source_data_inventory"])
            changed_dataset = deepcopy(current)
            changed_dataset["source_contract"]["dataset_artifacts"] = {"feature_bank": "changed"}
            check_true("reverse_bridge_dataset_sha_change_never_reused",
                       "source_contract.dataset_artifacts mismatch" in
                       owner._first_fold_fit_identity_issues(
                           historical, changed_dataset, project_root=root_dir,
                       ))
            # AI: Derive safe/unsafe capabilities from the canonical profile
            # registry, independent of the current comparison-model settings.
            import hashlib
            from core.breakout_quality_registry import (
                SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
                TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
            )
            from core.breakout_quality_runtime import (
                CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
                CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
            )
            from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
            from filters.breakout_quality.dataset_store import resolve_dataset_paths

            profiles = [
                p for p in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
                if get_breakout_quality_experiment_profile(p).training_sample_scope
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
            ]
            self_contained = next(
                (p for p in profiles
                 if get_continuous_ranker_execution_recipe(p).context_policy.source
                 == CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
                 and get_continuous_ranker_execution_recipe(p).target_policy.materialization_mode
                 == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT), None
            )
            external_context = next(
                (p for p in profiles
                 if get_continuous_ranker_execution_recipe(p).context_policy.source
                 != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE), None
            )
            check_true("reverse_bridge_self_contained_and_external_context_profiles_available",
                       self_contained is not None and external_context is not None)
            if self_contained is not None and external_context is not None:
                files = resolve_dataset_paths(Path(".")).artifact_paths()
                event_records = {
                    key: {
                        "filename": path.name,
                        "sha256": hashlib.sha256(f"old:{key}".encode()).hexdigest(),
                        "size_bytes": 10,
                    }
                    for key, path in files.items()
                }
                old_daily = deepcopy(historical)
                old_daily["training_sample_scope"] = TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                old_daily["experiment_profile"] = self_contained
                old_daily["source_contract"]["target_contract"] = {"source": "synthetic_daily_target"}
                old_daily["source_contract"]["dataset_artifacts"] = event_records
                new_daily = deepcopy(old_daily)
                new_daily["source_contract"]["source_data_inventory"] = deepcopy(
                    current["source_contract"]["source_data_inventory"]
                )
                new_daily["source_contract"]["dataset_artifacts"] = deepcopy(event_records)
                new_daily["source_contract"]["dataset_artifacts"]["event_labels"]["sha256"] = hashlib.sha256(
                    b"relabel-only-refresh"
                ).hexdigest()
                check("reverse_bridge_verified_daily_lazy_event_relabel_does_not_change_fit", (),
                      owner._first_fold_fit_identity_issues(old_daily, new_daily, project_root=root_dir))
                check_true("reverse_bridge_original_fitkey_not_rewritten_by_event_relabel",
                           owner.fold_training_identity(old_daily) != owner.fold_training_identity(new_daily))

                # AI: Native Daily-component target_manifest has no 'artifacts':
                # PIT persists target_artifacts=None. A synthetic SHA fixture
                # falsely passed while the actual native contract was rejected.
                lazy_target_old = deepcopy(old_daily)
                lazy_target_new = deepcopy(new_daily)
                lazy_target_old["source_contract"]["target_artifacts"] = None
                lazy_target_new["source_contract"]["target_artifacts"] = None
                check("reverse_bridge_real_daily_lazy_target_has_no_artifact_files", (),
                      owner._first_fold_fit_identity_issues(
                          lazy_target_old, lazy_target_new, project_root=root_dir,
                      ))
                # AI: Old native Rolling folds can explicitly publish None for
                # unused event artifacts. This is safe ONLY for a daily-component,
                # no-external-context fit with independently matched source CSV,
                # target contract, periods/cohort and complete current evidence.
                native_none_old = deepcopy(lazy_target_old)
                native_none_old["source_contract"]["dataset_artifacts"] = None
                check("reverse_bridge_legacy_none_event_artifacts_do_not_force_refit", (),
                      owner._first_fold_fit_identity_issues(
                          native_none_old, lazy_target_new, project_root=root_dir,
                      ))
                check_true("reverse_bridge_legacy_none_keeps_original_fitkey",
                           owner.fold_training_identity(native_none_old)
                           != owner.fold_training_identity(lazy_target_new))
                missing_historical_event_field = deepcopy(native_none_old)
                missing_historical_event_field["source_contract"].pop("dataset_artifacts")
                check_true("reverse_bridge_absent_event_field_is_not_explicit_none",
                           "source_contract.dataset_artifacts missing" in
                           owner._first_fold_fit_identity_issues(
                               missing_historical_event_field, lazy_target_new,
                               project_root=root_dir,
                           ))
                malformed_historical_event = deepcopy(native_none_old)
                malformed_historical_event["source_contract"]["dataset_artifacts"] = "legacy-unknown"
                check_true("reverse_bridge_arbitrary_historical_event_value_rejected",
                           "dataset_artifacts proof blocked: historical event Dataset "
                           "records have an unsupported representation" in
                           owner._first_fold_fit_identity_issues(
                               malformed_historical_event, lazy_target_new,
                               project_root=root_dir,
                           ))
                partial_historical_event = deepcopy(native_none_old)
                partial_historical_event["source_contract"]["dataset_artifacts"] = {}
                check_true("reverse_bridge_partial_historical_event_mapping_not_treated_as_none",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               partial_historical_event, lazy_target_new,
                               project_root=root_dir,
                           ))
                changed_csv_old = deepcopy(native_none_old)
                changed_csv_old["source_contract"]["source_data_inventory"] = None
                check_true("reverse_bridge_none_event_cannot_replace_missing_csv_inventory",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               changed_csv_old, lazy_target_new, project_root=root_dir,
                           ))
                changed_target_for_none = deepcopy(lazy_target_new)
                changed_target_for_none["source_contract"]["target_contract"] = {
                    "source": "different-target",
                }
                check_true("reverse_bridge_none_event_cannot_replace_target_contract",
                           "source_contract.target_contract mismatch" in
                           owner._first_fold_fit_identity_issues(
                               native_none_old, changed_target_for_none,
                               project_root=root_dir,
                           ))
                changed_samples_for_none = deepcopy(lazy_target_new)
                changed_samples_for_none["group_counts"]["inner_train"] += 1
                check_true("reverse_bridge_none_event_cannot_replace_sample_membership",
                           "group_counts.inner_train mismatch" in
                           owner._first_fold_fit_identity_issues(
                               native_none_old, changed_samples_for_none,
                               project_root=root_dir,
                           ))
                native_none_external_old = deepcopy(native_none_old)
                native_none_external_new = deepcopy(lazy_target_new)
                native_none_external_old["experiment_profile"] = external_context
                native_none_external_new["experiment_profile"] = external_context
                check_true("reverse_bridge_none_event_not_exempt_for_predicted_context",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               native_none_external_old, native_none_external_new,
                               project_root=root_dir,
                           ))
                missing_target_field = deepcopy(lazy_target_old)
                missing_target_field["source_contract"].pop("target_artifacts")
                check_true("reverse_bridge_unrecorded_target_artifact_contract_rejected",
                           "source_contract.target_artifacts missing" in
                           owner._first_fold_fit_identity_issues(
                               missing_target_field, lazy_target_new,
                               project_root=root_dir,
                           ))
                substituted_target = deepcopy(lazy_target_new)
                substituted_target["source_contract"]["target_artifacts"] = {
                    "target_file": {"sha256": hashlib.sha256(b"changed").hexdigest()}
                }
                check_true("reverse_bridge_lazy_target_new_physical_lineage_rejected",
                           "source_contract.target_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               lazy_target_old, substituted_target,
                               project_root=root_dir,
                           ))
                changed_lazy_target_contract = deepcopy(lazy_target_new)
                changed_lazy_target_contract["source_contract"]["target_contract"] = {
                    "source": "different_daily_target",
                }
                check_true("reverse_bridge_lazy_target_contract_change_rejected",
                           "source_contract.target_contract mismatch" in
                           owner._first_fold_fit_identity_issues(
                               lazy_target_old, changed_lazy_target_contract,
                               project_root=root_dir,
                           ))

                unsafe_context_old = deepcopy(old_daily)
                unsafe_context_new = deepcopy(new_daily)
                unsafe_context_old["experiment_profile"] = external_context
                unsafe_context_new["experiment_profile"] = external_context
                check_true("reverse_bridge_external_context_requires_original_dataset_lineage",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               unsafe_context_old, unsafe_context_new, project_root=root_dir))
                missing_dataset = deepcopy(old_daily)
                missing_dataset["source_contract"]["dataset_artifacts"].pop("events_csv")
                check_true("reverse_bridge_partial_dataset_hash_records_rejected",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               missing_dataset, new_daily, project_root=root_dir))
                # AI: The actual Windows checkpoint is not packaged with
                # source ZIPs. Report a field-level *blocker* when the bridge
                # refuses historical FIT evidence, rather than guessing a
                # scientific-compatible source format from synthetic fixtures.
                missing_dataset_issues = owner._first_fold_fit_identity_issues(
                    missing_dataset, new_daily, project_root=root_dir,
                )
                check_true("reverse_bridge_missing_event_key_diagnostic",
                           "dataset_artifacts proof blocked: historical event Dataset key set differs: "
                           "missing=events_csv, unexpected_count=0, record_count=9"
                           in missing_dataset_issues)
                check_true("reverse_bridge_no_raw_hash_exposed_by_event_diagnostic",
                           all("sha-old" not in str(issue) and
                               event_records["feature_bank"]["sha256"] not in str(issue)
                               for issue in missing_dataset_issues))
                missing_current_records = deepcopy(new_daily)
                missing_current_records["source_contract"]["dataset_artifacts"] = None
                check_true("reverse_bridge_missing_current_event_records_explained",
                           "dataset_artifacts proof blocked: current event Dataset records "
                           "absent or not a mapping" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, missing_current_records, project_root=root_dir,
                           ))
                unsafe_context_issues = owner._first_fold_fit_identity_issues(
                    unsafe_context_old, unsafe_context_new, project_root=root_dir,
                )
                check_true("reverse_bridge_external_context_exemption_blocker_explained",
                           "dataset_artifacts proof blocked: external context requires "
                           "published context/fit binding proof" in unsafe_context_issues)
                bad_dataset = deepcopy(new_daily)
                bad_dataset["source_contract"]["dataset_artifacts"]["feature_bank"]["sha256"] = "not-a-sha"
                check_true("reverse_bridge_unverifiable_dataset_sha_rejected",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, bad_dataset, project_root=root_dir))
                bad_dataset_issues = owner._first_fold_fit_identity_issues(
                    old_daily, bad_dataset, project_root=root_dir,
                )
                check_true("reverse_bridge_invalid_event_sha_diagnostic",
                           "dataset_artifacts proof blocked: current feature_bank SHA256 "
                           "metadata invalid" in bad_dataset_issues)
                missing_inventory_proof = deepcopy(old_daily)
                missing_inventory_proof["source_contract"]["source_data_inventory"] = None
                matching_missing_inventory = deepcopy(new_daily)
                matching_missing_inventory["source_contract"]["source_data_inventory"] = None
                check_true("reverse_bridge_missing_inventory_records_not_a_proof",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               missing_inventory_proof, matching_missing_inventory,
                               project_root=root_dir))
                absent_target_contract = deepcopy(new_daily)
                absent_target_contract["source_contract"]["target_contract"] = None
                check_true("reverse_bridge_missing_target_contract_not_a_proof",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, absent_target_contract, project_root=root_dir))
                absent_target_hashes = deepcopy(new_daily)
                absent_target_hashes["source_contract"]["target_artifacts"] = None
                check_true("reverse_bridge_missing_target_hashes_not_a_proof",
                           "source_contract.dataset_artifacts mismatch" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, absent_target_hashes, project_root=root_dir))
                changed_source = deepcopy(new_daily)
                changed_source["source_contract"]["input_price_source"] = {"materialization_fingerprint": "new"}
                check_true("reverse_bridge_supplemental_feature_source_change_rejected",
                           "source_contract.input_price_source missing" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, changed_source, project_root=root_dir))
                changed_training = deepcopy(new_daily)
                changed_training["group_counts"]["inner_train"] += 1
                check_true("reverse_bridge_new_dataset_does_not_mask_training_cohort_change",
                           "group_counts.inner_train mismatch" in
                           owner._first_fold_fit_identity_issues(
                               old_daily, changed_training, project_root=root_dir))

            changed_target = deepcopy(current)
            changed_target["source_contract"]["target_artifacts"] = {"raw_target": "changed"}
            check_true("reverse_bridge_target_sha_change_never_reused",
                       "source_contract.target_artifacts mismatch" in
                       owner._first_fold_fit_identity_issues(
                           historical, changed_target, project_root=root_dir,
                       ))
            missing_inventory = deepcopy(historical)
            missing_inventory["source_contract"].pop("source_data_inventory")
            check_true("reverse_bridge_missing_inventory_never_reused",
                       "source_contract.source_data_inventory missing" in
                       owner._first_fold_fit_identity_issues(
                           missing_inventory, current, project_root=root_dir,
                       ))
            changed_seed = deepcopy(current)
            changed_seed["seed"] = 314
            check_true("reverse_bridge_seed_change_never_reused",
                       "seed mismatch" in owner._first_fold_fit_identity_issues(
                           historical, changed_seed, project_root=root_dir,
                       ))

            # AI: Validate native checkpoint bytes and embedded Fold identity
            # without requiring a GPU training run.
            from filters.breakout_quality.artifacts import build_file_manifest
            fold_dir = root_dir / "fold_20210101_20211231"
            fold_dir.mkdir()
            checkpoint_path = fold_dir / "model.pt"
            checkpoint_path.write_bytes(b"archived-synthetic-checkpoint")
            manifest_path = fold_dir / "manifest.json"
            execution = {"device_type": "cpu", "deterministic_algorithms": True}
            manifest = {
                **historical, "fold_id": fold_dir.name,
                "torch_execution": {**execution, "requested_device": "auto"},
                "selected_epoch": 2, "epoch_selection": {"best_epoch": 2},
                "artifacts": {"checkpoint": build_file_manifest(checkpoint_path)},
            }
            checkpoint = {
                "model_spec": historical["model_spec"],
                "experiment_settings": historical["experiment_settings"],
                "selected_epoch": 2, "fold_contract": historical,
            }
            bundle = SimpleNamespace(
                model_spec=SimpleNamespace(as_manifest_payload=lambda: historical["model_spec"]),
                profile=SimpleNamespace(as_manifest_payload=lambda: historical["experiment_settings"]),
            )
            verify_args = dict(
                manifest_path=manifest_path, manifest=manifest, expected=current,
                expected_execution=execution, bundle=bundle,
                torch_module=SimpleNamespace(load=lambda *_args, **_kwargs: checkpoint),
                project_root=root_dir,
            )
            candidate, reason = owner.validate_rolling_fold_fitted_checkpoint(**verify_args)
            check_true("reverse_bridge_verified_fold_bytes_and_embedded_identity",
                       candidate is not None and reason is None
                       and candidate.selected_epoch == 2)
            if self_contained is not None and external_context is not None:
                # AI: Test native manifest and embedded checkpoint through FIT owner.
                verified_daily_manifest = {**manifest, **old_daily}
                verified_daily_checkpoint = {
                    **checkpoint, "fold_contract": old_daily,
                    "experiment_settings": old_daily["experiment_settings"],
                    "model_spec": old_daily["model_spec"],
                }
                verified_daily, verified_reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                    **verify_args, "manifest": verified_daily_manifest, "expected": new_daily,
                    "torch_module": SimpleNamespace(load=lambda *_args, **_kwargs: verified_daily_checkpoint),
                })
                check_true("reverse_bridge_complete_fold_checkpoint_allows_proven_event_artifact_refresh",
                           verified_daily is not None and verified_reason is None)
                # AI: Exercise both outer native manifest and checkpoint-embedded
                # contract with actual lazy-target None, not a fabricated SHA.
                verified_lazy_manifest = {**manifest, **lazy_target_old}
                verified_lazy_checkpoint = {
                    **checkpoint, "fold_contract": lazy_target_old,
                    "experiment_settings": lazy_target_old["experiment_settings"],
                    "model_spec": lazy_target_old["model_spec"],
                }
                verified_lazy, verified_lazy_reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                    **verify_args, "manifest": verified_lazy_manifest, "expected": lazy_target_new,
                    "torch_module": SimpleNamespace(load=lambda *_args, **_kwargs: verified_lazy_checkpoint),
                })
                check_true("reverse_bridge_native_lazy_target_fold_and_checkpoint_reuse",
                           verified_lazy is not None and verified_lazy_reason is None)
                # AI: Reproduce the reported Windows failure shape through
                # both the published Fold manifest and the checkpoint's embedded
                # contract; a helper-only check can miss the second validation.
                verified_none_manifest = {**manifest, **native_none_old}
                verified_none_checkpoint = {
                    **checkpoint, "fold_contract": native_none_old,
                    "experiment_settings": native_none_old["experiment_settings"],
                    "model_spec": native_none_old["model_spec"],
                }
                verified_none, verified_none_reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                    **verify_args, "manifest": verified_none_manifest,
                    "expected": lazy_target_new,
                    "torch_module": SimpleNamespace(load=lambda *_args, **_kwargs: verified_none_checkpoint),
                })
                check_true("reverse_bridge_native_none_event_manifest_and_checkpoint_reuse",
                           verified_none is not None and verified_none_reason is None)
                # The dataset exemption must not bypass checkpoint integrity.
                damaged_none_manifest = deepcopy(verified_none_manifest)
                damaged_none_manifest["artifacts"]["checkpoint"]["sha256"] = "0" * 64
                try:
                    owner.validate_rolling_fold_fitted_checkpoint(**{
                        **verify_args, "manifest": damaged_none_manifest,
                        "expected": lazy_target_new,
                        "torch_module": SimpleNamespace(load=lambda *_args, **_kwargs: verified_none_checkpoint),
                    })
                except RuntimeError as exc:
                    hash_rejected = "checkpoint hash mismatch" in str(exc)
                else:
                    hash_rejected = False
                check_true("reverse_bridge_native_none_event_checkpoint_hash_still_enforced", hash_rejected)
            # AI: Generic external-context policy, selected from the registry.
            # Independently link original PIT aggregate -> Fold SHA -> checkpoint
            # and recorded context bytes. Both outer and embedded native contracts
            # must pass; current context READY alone is explicitly insufficient.
            if external_context is not None:
                import json
                from filters.breakout_quality.paths import SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
                from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec
                context_recipe = get_continuous_ranker_execution_recipe(external_context)
                context_spec = maybe_predicted_context_artifact_spec(context_recipe.context_policy.source)
                check_true("ssot_external_context_has_registered_artifact_capability", context_spec is not None)
                if context_spec is not None:
                    native_context_old = deepcopy(native_none_old)
                    native_context_new = deepcopy(lazy_target_new)
                    for item in (native_context_old, native_context_new):
                        item["experiment_profile"] = external_context
                        # A real registered architecture is needed when resolving
                        # the original Forward namespace for migrated FIT proof.
                        item["model_architecture"] = model_settings.model_architecture
                        item["training_settings"] = {
                            "epochs_max": 3, "batch_size": 128,
                            "learning_rate": 0.001, "weight_decay": 0.0001,
                            "gradient_clip_norm": 1.0,
                            "early_stopping_patience": 1,
                            "early_stopping_min_delta": 0.0,
                        }
                    pit_root = root_dir / "published_pit"
                    ctx_fold_dir = pit_root / "folds" / "fold_20210101_20211231"
                    ctx_fold_dir.mkdir(parents=True)
                    ctx_ckpt = ctx_fold_dir / "model.pt"
                    ctx_ckpt.write_bytes(b"context_checkpoint_synthetic")
                    ctx_path = ctx_fold_dir / "manifest.json"
                    ctx_manifest = {
                        **native_context_old, "fold_id": ctx_fold_dir.name,
                        "torch_execution": dict(execution),
                        "selected_epoch": 2, "epoch_selection": {"best_epoch": 2},
                        "artifacts": {"checkpoint": build_file_manifest(ctx_ckpt)},
                    }
                    ctx_path.write_text(json.dumps(ctx_manifest), encoding="utf-8")
                    ctx_embedded = {
                        "model_spec": native_context_old["model_spec"],
                        "experiment_settings": native_context_old["experiment_settings"],
                        "selected_epoch": 2, "fold_contract": native_context_old,
                    }
                    stage = {
                        "research_id": "synthetic_s1", "profile": "stage1_profile",
                        "architecture": "stage1_arch", "seed": 42,
                        **{k: {"sha256": "a" * 64, "size_bytes": 10} for k in (
                            "selection_score", "selection_manifest",
                            "forward_score", "forward_manifest",
                        )},
                    }
                    context_record = {
                        "schema_version": 1, "filter_id": "synthetic",
                        "model_architecture": "stage1_arch",
                        "experiment_profile": "stage1_profile", "contract": {"pit": True},
                        "dataset_policy": "canonical", "periods": {"oos_start": "2021-01-01"},
                        "rows": {"total": 50},
                        "artifact": {"sha256": "b" * 64, "size_bytes": 123},
                        "source_stage1": stage,
                    }
                    target_lineage = {
                        "schema_version": None,
                        "target_contract": native_context_old["source_contract"]["target_contract"],
                        "artifacts": None,
                        context_spec.manifest_key: context_record,
                    }
                    ctx_bundle = SimpleNamespace(
                        model_spec=bundle.model_spec, profile=bundle.profile,
                        target_manifest=target_lineage,
                    )
                    aggregate = {
                        **{k: ctx_manifest[k] for k in (
                            "filter_id", "model_architecture", "experiment_profile",
                            "continuous_target_id", "training_sample_scope",
                            "training_label_scope", "seed",
                        )},
                        "folds": [{
                            "fold_id": ctx_manifest["fold_id"],
                            "fold_manifest": build_file_manifest(ctx_path),
                            "artifacts": ctx_manifest["artifacts"],
                        }],
                        "source_continuous_target": target_lineage,
                    }
                    aggregate_path = pit_root / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    ctx_args = {
                        **verify_args, "manifest_path": ctx_path,
                        "manifest": ctx_manifest, "expected": native_context_new,
                        "bundle": ctx_bundle,
                        "torch_module": SimpleNamespace(load=lambda *_a, **_k: ctx_embedded),
                    }
                    # The opposite Forward -> Rolling bridge consumes exactly
                    # the same snapshot projection and registered Context proof.
                    from services.breakout_quality.fitting_data_dependencies import (
                        forward_publication_fit_dependency_issues,
                    )
                    published_dataset = {
                        "policy": native_context_new["source_contract"]["dataset_policy"],
                        "source_data_inventory": native_context_new["source_contract"]["source_data_inventory"],
                        "source_dataset_artifacts": native_context_new["source_contract"]["dataset_artifacts"],
                    }
                    opposite_kwargs = dict(
                        source_dataset=published_dataset, source_target=target_lineage,
                        expected_fold_source=native_context_new["source_contract"],
                        current_target=target_lineage,
                        experiment_profile=external_context,
                        project_root=root_dir,
                        inventory_matcher=lambda *, published, current, project_root:
                            published == current,
                    )
                    check("ssot_forward_to_rolling_reads_same_declared_context", (),
                          forward_publication_fit_dependency_issues(**opposite_kwargs))
                    changed_published_target = deepcopy(target_lineage)
                    changed_published_target[context_spec.manifest_key]["artifact"]["sha256"] = "d" * 64
                    check_true("ssot_forward_to_rolling_context_bytes_drift_rejected",
                               "source_context proof blocked" in str(
                                   forward_publication_fit_dependency_issues(**{
                                       **opposite_kwargs, "source_target": changed_published_target,
                                   })
                               ))
                    changed_stage_provenance = deepcopy(target_lineage)
                    changed_stage_provenance[context_spec.manifest_key]["source_stage1"]["forward_manifest"]["sha256"] = "e" * 64
                    check_true("ssot_forward_to_rolling_stage1_artifact_drift_rejected",
                               "Stage-1 forward_manifest artifact lineage differs" in str(
                                   forward_publication_fit_dependency_issues(**{
                                       **opposite_kwargs, "source_target": changed_stage_provenance,
                                   })
                               ))
                    replaced_target_contract = deepcopy(target_lineage)
                    replaced_target_contract["target_contract"] = {"synthetic": "changed"}
                    check_true("ssot_forward_to_rolling_target_semantic_drift_rejected",
                               "source_contract target_contract mismatch" in
                               forward_publication_fit_dependency_issues(**{
                                   **opposite_kwargs, "source_target": replaced_target_contract,
                               }))
                    changed_published_dataset = deepcopy(published_dataset)
                    changed_published_dataset["source_data_inventory"] = {"source": "modified"}
                    check_true("ssot_forward_to_rolling_source_csv_drift_rejected",
                               "source_contract source_data_inventory mismatch" in
                               forward_publication_fit_dependency_issues(**{
                                   **opposite_kwargs, "source_dataset": changed_published_dataset,
                               }))
                    check_true("ssot_forward_to_rolling_unregistered_context_recipe_rejected",
                               "unregistered fitting context recipe" in
                               forward_publication_fit_dependency_issues(**{
                                   **opposite_kwargs, "experiment_profile": "unregistered_profile",
                               }))
                    ok_ctx, ctx_reason = owner.validate_rolling_fold_fitted_checkpoint(**ctx_args)
                    check_true("ssot_published_external_context_both_native_fit_contracts_reusable",
                               ok_ctx is not None and ctx_reason is None)
                    bad_ctx = deepcopy(target_lineage)
                    bad_ctx[context_spec.manifest_key]["artifact"]["sha256"] = "c" * 64
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(
                        **{**ctx_args, "bundle": SimpleNamespace(
                            model_spec=bundle.model_spec, profile=bundle.profile,
                            target_manifest=bad_ctx,
                        )},
                    )
                    check_true("ssot_changed_context_content_blocks_even_with_other_fitting_inputs_equal",
                               denied is None and "external fitting context proof blocked" in str(reason))
                    altered_stage = deepcopy(target_lineage)
                    altered_stage[context_spec.manifest_key]["source_stage1"]["selection_score"]["sha256"] = "f" * 64
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(
                        **{**ctx_args, "bundle": SimpleNamespace(
                            model_spec=bundle.model_spec, profile=bundle.profile,
                            target_manifest=altered_stage,
                        )},
                    )
                    check_true("ssot_rolling_to_forward_stage1_artifact_drift_rejected",
                               denied is None and "Stage-1 selection_score artifact lineage differs" in str(reason))
                    unchanged_sources = deepcopy(native_context_old)
                    unchanged_sources["source_contract"]["dataset_artifacts"] = deepcopy(
                        native_context_new["source_contract"]["dataset_artifacts"]
                    )
                    equal_manifest = {**ctx_manifest, "source_contract": unchanged_sources["source_contract"]}
                    equal_manifest["artifacts"] = deepcopy(ctx_manifest["artifacts"])
                    equal_ckpt = {**ctx_embedded, "fold_contract": unchanged_sources}
                    # Rebind published fold to this variant so event Dataset
                    # equality cannot mask an unproven context input change.
                    ctx_path.write_text(json.dumps(equal_manifest), encoding="utf-8")
                    aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                        **ctx_args, "manifest": equal_manifest, "expected": native_context_new,
                        "bundle": SimpleNamespace(
                            model_spec=bundle.model_spec, profile=bundle.profile,
                            target_manifest=bad_ctx,
                        ),
                        "torch_module": SimpleNamespace(load=lambda *_a, **_k: equal_ckpt),
                    })
                    check_true("ssot_same_event_sha_cannot_bypass_external_context_identity",
                               denied is None and "external fitting context proof blocked" in str(reason))
                    # Restore native original and its aggregate SHA.
                    ctx_path.write_text(json.dumps(ctx_manifest), encoding="utf-8")
                    aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    aggregate_bad = deepcopy(aggregate)
                    aggregate_bad["folds"][0]["fold_manifest"]["sha256"] = "0" * 64
                    aggregate_path.write_text(json.dumps(aggregate_bad), encoding="utf-8")
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(**ctx_args)
                    check_true("ssot_published_fold_sha_tamper_blocks_context_reuse",
                               denied is None and "fold/checkpoint SHA" in str(reason))
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    # AI: A post-rescore migration changes Fold manifest bytes.
                    # The aggregate must be rebased for an actual provenance test.
                    migrated_manifest = {**ctx_manifest, "migration": {"kind": "copy"}}
                    ctx_path.write_text(json.dumps(migrated_manifest), encoding="utf-8")
                    aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                        **ctx_args, "manifest": migrated_manifest,
                    })
                    check_true("ssot_migrated_checkpoint_without_original_context_proof_is_blocked",
                               denied is None and "migrated Rolling checkpoint" in str(reason))
                    # Legacy Rolling-native checkpoints may later acquire a
                    # score-refresh migration. The refresh is NOT a Forward
                    # import: native checkpoint contract + execution must be
                    # checked against the current FIT, with aggregate and
                    # Context SHA still binding the published Fold bytes.
                    # This reproduces the historical second-fold failure where
                    # no original Forward publication is supposed to exist.
                    score_only_native = {
                        **ctx_manifest,
                        "migration": {
                            "kind": "expanded_daily_score_universe_checkpoint_reuse",
                            "training_contract_unchanged": True,
                            "source_checkpoint_sha_preserved": True,
                        },
                    }
                    native_fitted_checkpoint = {
                        **ctx_embedded, "torch_execution": dict(execution),
                    }
                    def verify_native_score_only(
                        candidate, checkpoint_to_check=native_fitted_checkpoint,
                        current_target=target_lineage,
                    ):
                        ctx_path.write_text(json.dumps(candidate), encoding="utf-8")
                        aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                        return owner.validate_rolling_fold_fitted_checkpoint(**{
                            **ctx_args, "manifest": candidate,
                            "torch_module": SimpleNamespace(
                                load=lambda *_a, **_k: checkpoint_to_check),
                            "bundle": SimpleNamespace(
                                model_spec=bundle.model_spec,
                                profile=bundle.profile,
                                target_manifest=current_target,
                            ),
                        })
                    reused_native, native_reason = verify_native_score_only(score_only_native)
                    check_true("ssot_native_rolling_score_refresh_does_not_require_nonexistent_forward_source",
                               reused_native is not None and native_reason is None)
                    copied_from_native = deepcopy(score_only_native)
                    copied_from_native["migration"] = {
                        "kind": "fitting_identity_checkpoint_reuse",
                        "training_contract_unchanged": True,
                        "source_checkpoint_sha_preserved": True,
                        "source_checkpoint": deepcopy(ctx_manifest["artifacts"]["checkpoint"]),
                        "initial_fitting_migration": deepcopy(score_only_native["migration"]),
                    }
                    reused_copy, copied_reason = verify_native_score_only(copied_from_native)
                    check_true("ssot_native_score_only_fold_cache_copy_preserves_reuse",
                               reused_copy is not None and copied_reason is None)
                    # An owner-verified legacy event-cache drift must be
                    # consumable by archive score regeneration too. The
                    # ordinary cache/score path must NOT acquire that bypass.
                    from services.breakout_quality.fitted_model_artifacts import (
                        _checkpoint_matches_rescore_contract,
                        _fold_training_contract_is_compatible,
                    )
                    shape_bundle = SimpleNamespace(
                        feature_bank=SimpleNamespace(shape=(1, 300, 10)),
                        group_context=SimpleNamespace(shape=(1, 1)),
                    )
                    shaped_checkpoint = {
                        **native_fitted_checkpoint,
                        "sequence_length": 300, "feature_count": 10,
                        "context_count": 1,
                    }
                    exact_old_schema = _fold_training_contract_is_compatible(
                        native_context_old, expected_contract=native_context_new,
                    )
                    old_schema_denied = _checkpoint_matches_rescore_contract(
                        checkpoint=shaped_checkpoint, manifest=score_only_native,
                        fold_contract=native_context_new, bundle=shape_bundle,
                    )
                    owner_verified_archive = _checkpoint_matches_rescore_contract(
                        checkpoint=shaped_checkpoint, manifest=score_only_native,
                        fold_contract=native_context_new, bundle=shape_bundle,
                        verified_archived_fit=True, project_root=root_dir,
                    )
                    check_true("ssot_archive_reuses_canonical_proven_nonconsumed_event_artifact_drift",
                               (not exact_old_schema) and (not old_schema_denied)
                               and owner_verified_archive)
                    different_seed = deepcopy(native_context_new)
                    different_seed["seed"] = 777
                    check_true("ssot_archive_verified_flag_does_not_bypass_changed_seed",
                               not _checkpoint_matches_rescore_contract(
                                   checkpoint=shaped_checkpoint, manifest=score_only_native,
                                   fold_contract=different_seed, bundle=shape_bundle,
                                   verified_archived_fit=True, project_root=root_dir,
                               ))
                    no_native_contract = deepcopy(native_fitted_checkpoint)
                    no_native_contract.pop("fold_contract")
                    blocked_native, blocked_reason = verify_native_score_only(
                        score_only_native, checkpoint_to_check=no_native_contract)
                    check_true("ssot_legacy_score_refresh_missing_embedded_fold_contract_blocks",
                               blocked_native is None and "migrated Rolling checkpoint" in str(blocked_reason))
                    wrong_execution = deepcopy(native_fitted_checkpoint)
                    wrong_execution["torch_execution"]["deterministic_algorithms"] = False
                    blocked_native, blocked_reason = verify_native_score_only(
                        score_only_native, checkpoint_to_check=wrong_execution)
                    check_true("ssot_legacy_score_refresh_wrong_checkpoint_execution_blocks",
                               blocked_native is None and "migrated Rolling checkpoint" in str(blocked_reason))
                    wrong_fold = deepcopy(native_fitted_checkpoint)
                    wrong_fold["fold_contract"]["seed"] = 777
                    blocked_native, blocked_reason = verify_native_score_only(
                        score_only_native, checkpoint_to_check=wrong_fold)
                    check_true("ssot_legacy_score_refresh_wrong_embedded_fitkey_blocks",
                               blocked_native is None and "migrated Rolling checkpoint" in str(blocked_reason))
                    explicit_forward_import = deepcopy(score_only_native)
                    explicit_forward_import["migration"]["initial_fitting_migration"] = {
                        "kind": "forward_model_fitting_identity_import",
                        "training_contract_unchanged": True,
                        "source_checkpoint_sha_preserved": True,
                        "source_forward_checkpoint": deepcopy(ctx_manifest["artifacts"]["checkpoint"]),
                    }
                    blocked_native, blocked_reason = verify_native_score_only(explicit_forward_import)
                    check_true("ssot_forward_origin_marker_must_not_be_rebranded_native_rolling",
                               blocked_native is None and "migrated Rolling checkpoint" in str(blocked_reason))
                    invalid_source_sha = deepcopy(copied_from_native)
                    invalid_source_sha["migration"]["source_checkpoint"]["sha256"] = "0" * 64
                    blocked_native, blocked_reason = verify_native_score_only(invalid_source_sha)
                    check_true("ssot_native_score_only_bad_source_sha_blocks",
                               blocked_native is None and "migrated Rolling checkpoint" in str(blocked_reason))
                    changed_original_context = deepcopy(target_lineage)
                    changed_original_context[context_spec.manifest_key]["artifact"]["sha256"] = "c" * 64
                    blocked_native, blocked_reason = verify_native_score_only(
                        score_only_native, current_target=changed_original_context)
                    check_true("ssot_native_score_refresh_context_content_change_still_blocks",
                               blocked_native is None and "external fitting context proof blocked" in str(blocked_reason))
                    wrong_aggregate = deepcopy(aggregate)
                    wrong_aggregate["folds"][0]["fold_manifest"]["sha256"] = "0" * 64
                    ctx_path.write_text(json.dumps(score_only_native), encoding="utf-8")
                    aggregate_path.write_text(json.dumps(wrong_aggregate), encoding="utf-8")
                    blocked_native, blocked_reason = owner.validate_rolling_fold_fitted_checkpoint(**{
                        **ctx_args, "manifest": score_only_native,
                        "torch_module": SimpleNamespace(
                            load=lambda *_a, **_k: native_fitted_checkpoint),
                    })
                    check_true("ssot_native_score_refresh_aggregate_sha_drift_blocks",
                               blocked_native is None and "fold/checkpoint SHA" in str(blocked_reason))
                    # Restore the normal test fixture before Forward-origin cases.
                    ctx_path.write_text(json.dumps(ctx_manifest), encoding="utf-8")
                    aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    
                    from filters.breakout_quality.paths import (
                        resolve_filter_artifact_paths, resolve_filter_model_output_dir,
                    )
                    model_paths = resolve_filter_artifact_paths(
                        root_dir, ctx_manifest["filter_id"],
                        ctx_manifest["model_architecture"], external_context,
                    )
                    forward_output = resolve_filter_model_output_dir(
                        root_dir, ctx_manifest["filter_id"],
                        ctx_manifest["model_architecture"], external_context,
                    )
                    model_paths.model_dir.mkdir(parents=True, exist_ok=True)
                    forward_output.mkdir(parents=True, exist_ok=True)
                    model_paths.model_path.write_bytes(ctx_ckpt.read_bytes())
                    forward_checkpoint_record = build_file_manifest(model_paths.model_path)
                    source_report = {
                        "training": {
                            "seed": ctx_manifest["seed"],
                            "selected_epoch": ctx_manifest["selected_epoch"],
                            "epoch_selection": {"best_epoch": ctx_manifest["selected_epoch"]},
                            "final_refit_history": [{"epoch": ctx_manifest["selected_epoch"]}],
                        },
                        "split_report": {
                            "inner_validation_start_date": ctx_manifest["planned_periods"]["validation_start"],
                            "selection_end_date": ctx_manifest["planned_periods"]["validation_end"],
                            "oos_start_date": ctx_manifest["planned_periods"]["score_start"],
                            "counts": {
                                "inner_train": ctx_manifest["group_counts"]["inner_train"],
                                "validation": ctx_manifest["group_counts"]["validation"],
                                "selection": ctx_manifest["group_counts"]["final_refit"],
                            },
                        },
                        "standard_model_sop": {"evaluation_mode": "forward_oos"},
                        "artifacts": {"model": forward_checkpoint_record},
                    }
                    report_path = forward_output / "continuous_ranker_report.json"
                    report_path.write_text(json.dumps(source_report), encoding="utf-8")
                    source_manifest = {
                        **{k: ctx_manifest[k] for k in (
                            "filter_id", "model_architecture", "experiment_profile",
                            "continuous_target_id", "training_label_scope", "training_sample_scope",
                            "model_spec", "experiment_settings", "model_information_cutoff",
                            "selected_epoch",
                        )},
                        "model": forward_checkpoint_record,
                        "research_outputs": {"report_json": build_file_manifest(report_path)},
                        "source_dataset": {
                            "policy": ctx_manifest["source_contract"]["dataset_policy"],
                            "source_data_inventory": ctx_manifest["source_contract"]["source_data_inventory"],
                        },
                        "source_continuous_target": target_lineage,
                    }
                    model_paths.manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
                    fitted_settings = {
                        "max_epochs": ctx_manifest["training_settings"]["epochs_max"],
                        **{key: ctx_manifest["training_settings"][key] for key in (
                            "batch_size", "learning_rate", "weight_decay",
                            "gradient_clip_norm", "early_stopping_patience",
                            "early_stopping_min_delta",
                        )},
                        "use_inner_validation": True,
                        "inner_validation_months": 24,
                    }
                    fitted_sidecar_path = model_paths.model_dir / "fitted_model_manifest.json"
                    fitted_sidecar = {
                        "contract_version": 1,
                        "fitting_identity": {
                            "training_settings": fitted_settings,
                            "torch_execution": execution,
                        },
                        "fitting_result": {"selected_epoch": ctx_manifest["selected_epoch"]},
                        "model": forward_checkpoint_record,
                    }
                    fitted_sidecar_path.write_text(json.dumps(fitted_sidecar), encoding="utf-8")
                    # AI: A valid legacy Forward-import fitting cache can be
                    # upgraded *only while the original verified Forward bytes
                    # still exist*. Previously the importer returned early and
                    # silently skipped the producer-captured origin witness.
                    from services.breakout_quality import point_in_time_scores as pit_owner
                    from shutil import copy2
                    cache_root = root_dir / "isolated_fit_cache"
                    entry = pit_owner._checkpoint_cache_entry_dir(cache_root, ctx_manifest)
                    entry.mkdir(parents=True)
                    copy2(model_paths.model_path, entry / "model.pt")
                    legacy_cache = {
                        **ctx_manifest,
                        "migration": {
                            "kind": "forward_model_fitting_identity_import",
                            "training_contract_unchanged": True,
                            "source_forward_checkpoint": forward_checkpoint_record,
                        },
                    }
                    cache_manifest_path = entry / "manifest.json"
                    cache_manifest_path.write_text(json.dumps(legacy_cache), encoding="utf-8")
                    original_cache_checkpoint_sha = build_file_manifest(entry / "model.pt")
                    fake_evidence = SimpleNamespace(
                        training_settings=fitted_settings,
                        torch_execution=execution,
                        selected_epoch=ctx_manifest["selected_epoch"],
                    )
                    def run_verified_cache_import():
                        # Input identity checks are exercised above through the
                        # canonical validator; isolate the post-proof cache path.
                        with patch.object(pit_owner, "_forward_checkpoint_import_issues", return_value=[]), \
                             patch.object(pit_owner, "load_fitted_model_training_evidence", return_value=(fake_evidence, None)):
                            return pit_owner._import_forward_checkpoint_to_fitting_cache(
                                source_model_dir=model_paths.model_dir,
                                source_report_path=report_path,
                                cache_root=cache_root,
                                fold_contract=ctx_manifest,
                                bundle=SimpleNamespace(),
                                torch_module=SimpleNamespace(load=lambda *_a, **_k: {"version": 1}),
                                execution_plan=SimpleNamespace(),
                            )
                    upgraded_entry = run_verified_cache_import()
                    upgraded_cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
                    check_true("ssot_existing_forward_import_cache_captures_verified_original_context",
                               upgraded_entry is not None
                               and upgraded_entry["reused_existing_cache"]
                               and upgraded_cache["fitting_origin"]["kind"] == "forward_import"
                               and upgraded_cache["fitting_origin"]["checkpoint"] == forward_checkpoint_record
                               and build_file_manifest(entry / "model.pt") == original_cache_checkpoint_sha)
                    upgraded_hash = build_file_manifest(cache_manifest_path)
                    run_verified_cache_import()
                    check("ssot_verified_cache_enrichment_is_idempotent", upgraded_hash,
                          build_file_manifest(cache_manifest_path))
                    tampered = deepcopy(upgraded_cache)
                    tampered["fitting_origin"]["checkpoint"]["sha256"] = "0" * 64
                    cache_manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
                    before_conflict = build_file_manifest(cache_manifest_path)
                    try:
                        run_verified_cache_import()
                    except RuntimeError as exc:
                        conflicting_origin_blocked = "conflicting Forward context origin" in str(exc)
                    else:
                        conflicting_origin_blocked = False
                    check_true("ssot_cached_forward_origin_conflict_blocks_without_mutation",
                               conflicting_origin_blocked
                               and before_conflict == build_file_manifest(cache_manifest_path))
                    native_cache = {
                        **legacy_cache,
                        "migration": {"kind": "rolling_native"},
                    }
                    cache_manifest_path.write_text(json.dumps(native_cache), encoding="utf-8")
                    native_hash = build_file_manifest(cache_manifest_path)
                    run_verified_cache_import()
                    check("ssot_native_rolling_cache_never_rebranded_forward_import", native_hash,
                          build_file_manifest(cache_manifest_path))
                    invalid_legacy = {**legacy_cache, "selected_epoch": ctx_manifest["selected_epoch"] + 1}
                    cache_manifest_path.write_text(json.dumps(invalid_legacy), encoding="utf-8")
                    try:
                        run_verified_cache_import()
                    except RuntimeError as exc:
                        epoch_conflict_blocked = "conflicting original Forward evidence" in str(exc)
                    else:
                        epoch_conflict_blocked = False
                    check_true("ssot_historical_forward_cache_epoch_conflict_blocks", epoch_conflict_blocked)
                    valid_migration = {
                        "kind": "fitting_identity_checkpoint_reuse",
                        "training_contract_unchanged": True,
                        "source_checkpoint_sha_preserved": True,
                        "source_checkpoint": deepcopy(ctx_manifest["artifacts"]["checkpoint"]),
                    }
                    linked_migration = {**ctx_manifest, "migration": valid_migration}
                    def verify_linked(manifest_to_check, target_to_check=target_lineage,
                                      checkpoint_to_check=ctx_embedded):
                        ctx_path.write_text(json.dumps(manifest_to_check), encoding="utf-8")
                        aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                        return owner.validate_rolling_fold_fitted_checkpoint(**{
                            **ctx_args, "manifest": manifest_to_check,
                            "bundle": SimpleNamespace(
                                model_spec=bundle.model_spec, profile=bundle.profile,
                                target_manifest=target_to_check,
                            ),
                            "torch_module": SimpleNamespace(
                                load=lambda *_a, **_k: checkpoint_to_check,
                            ),
                        })

                    ok, linked_reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_verified_forward_sha_origin_reusable",
                               ok is not None and linked_reason is None)
                    # The actual Forward imported model may have no native PIT
                    # fold_contract, and older rescore manifests may omit the
                    # selection decision. Both must come from proven Forward bytes.
                    no_native_epoch = deepcopy(linked_migration)
                    no_native_epoch.pop("epoch_selection", None)
                    imported_forward_checkpoint = deepcopy(ctx_embedded)
                    imported_forward_checkpoint.pop("fold_contract", None)
                    recovered, recovered_reason = verify_linked(
                        no_native_epoch, checkpoint_to_check=imported_forward_checkpoint,
                    )
                    check_true("ssot_migrated_forward_checkpoint_without_native_fold_uses_original_epoch",
                               recovered is not None and recovered_reason is None
                               and recovered.epoch_selection == source_report["training"]["epoch_selection"]
                               and recovered.final_refit_history == source_report["training"]["final_refit_history"])
                    bad_fitting = deepcopy(fitted_sidecar)
                    bad_fitting["fitting_identity"]["torch_execution"]["deterministic_algorithms"] = False
                    fitted_sidecar_path.write_text(json.dumps(bad_fitting), encoding="utf-8")
                    try:
                        verify_linked(no_native_epoch, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        execution_drift_blocked = "fitted-model torch_execution mismatch" in str(exc)
                    else:
                        execution_drift_blocked = False
                    check_true("ssot_migrated_forward_original_torch_execution_mismatch_blocks",
                               execution_drift_blocked)
                    fitted_sidecar_path.write_text(json.dumps(fitted_sidecar), encoding="utf-8")
                    bad_fitting = deepcopy(fitted_sidecar)
                    bad_fitting["model"]["sha256"] = "a" * 64
                    fitted_sidecar_path.write_text(json.dumps(bad_fitting), encoding="utf-8")
                    try:
                        verify_linked(no_native_epoch, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        wrong_model_blocked = "original fitted checkpoint SHA differs" in str(exc)
                    else:
                        wrong_model_blocked = False
                    check_true("ssot_migrated_forward_original_fit_sidecar_sha_mismatch_blocks",
                               wrong_model_blocked)
                    fitted_sidecar_path.write_text(json.dumps(fitted_sidecar), encoding="utf-8")
                    missing_epoch_report = deepcopy(source_report)
                    missing_epoch_report["training"].pop("epoch_selection", None)
                    report_path.write_text(json.dumps(missing_epoch_report), encoding="utf-8")
                    missing_epoch_source = deepcopy(source_manifest)
                    missing_epoch_source["research_outputs"]["report_json"] = build_file_manifest(report_path)
                    model_paths.manifest_path.write_text(json.dumps(missing_epoch_source), encoding="utf-8")
                    try:
                        verify_linked(no_native_epoch, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        epoch_gap_blocked = "lacks original epoch-selection evidence" in str(exc)
                    else:
                        epoch_gap_blocked = False
                    check_true("ssot_migrated_forward_no_native_epoch_and_no_source_epoch_blocks",
                               epoch_gap_blocked)
                    model_paths.manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
                    report_path.write_text(json.dumps(source_report), encoding="utf-8")
                    bad_source = deepcopy(source_manifest)
                    bad_source["source_continuous_target"][context_spec.manifest_key][
                        "source_stage1"]["selection_score"]["sha256"] = "f" * 64
                    model_paths.manifest_path.write_text(json.dumps(bad_source), encoding="utf-8")
                    denied, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_changed_forward_stage1_blocks",
                               denied is None and "original context binding" in str(reason))
                    model_paths.manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
                    bad_report = deepcopy(source_report)
                    bad_report["training"]["seed"] = 777
                    report_path.write_text(json.dumps(bad_report), encoding="utf-8")
                    denied, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_forward_report_identity_drift_blocks",
                               denied is None and "original context binding" in str(reason))
                    report_path.write_text(json.dumps(source_report), encoding="utf-8")
                    bad_link = deepcopy(linked_migration)
                    bad_link["migration"]["source_checkpoint"]["sha256"] = "0" * 64
                    denied, reason = verify_linked(bad_link)
                    check_true("ssot_migrated_context_source_checkpoint_drift_blocks",
                               denied is None and "source_checkpoint SHA" in str(reason))
                    unsafe_link = deepcopy(linked_migration)
                    unsafe_link["migration"]["training_contract_unchanged"] = False
                    denied, reason = verify_linked(unsafe_link)
                    check_true("ssot_migrated_context_unapproved_migration_blocks",
                               denied is None and "approved unchanged-checkpoint" in str(reason))
                    
                    # AI: The FIT owner preserves immutable original Forward
                    # evidence in historical_forward_fits when its publication
                    # is replaced by verified Rolling model bytes.
                    from shutil import copy2
                    archived = (model_paths.model_dir / "historical_forward_fits"
                                / forward_checkpoint_record["sha256"])
                    archived_model = archived / "model"
                    archived_report = archived / "report"
                    archived_model.mkdir(parents=True, exist_ok=True)
                    archived_report.mkdir(parents=True, exist_ok=True)
                    for fname in ("model.pt", "manifest.json", "fitted_model_manifest.json"):
                        copy2(model_paths.model_dir / fname, archived_model / fname)
                    copy2(report_path, archived_report / report_path.name)
                    preserved = [
                        {"path": relative, "manifest": build_file_manifest(path)}
                        for relative, path in (
                            ("model/model.pt", archived_model / "model.pt"),
                            ("model/manifest.json", archived_model / "manifest.json"),
                            ("model/fitted_model_manifest.json", archived_model / "fitted_model_manifest.json"),
                            ("report/continuous_ranker_report.json",
                             archived_report / "continuous_ranker_report.json"),
                        )
                    ]
                    archive_record = {
                        "archive_version": 1,
                        "original_model": forward_checkpoint_record,
                        "replacement_rolling_model": {"sha256": "9" * 64, "size_bytes": 10},
                        "preserved_files": preserved,
                    }
                    archive_path = archived / "archive_manifest.json"
                    archive_path.write_text(json.dumps(archive_record), encoding="utf-8")
                    model_paths.manifest_path.unlink()
                    model_paths.model_path.unlink()
                    fitted_sidecar_path.unlink()
                    report_path.unlink()
                    ok, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_verified_historical_forward_archive_reusable",
                               ok is not None and reason is None)
                    bad_archive = deepcopy(archive_record)
                    bad_archive["preserved_files"][1]["manifest"]["sha256"] = "0" * 64
                    archive_path.write_text(json.dumps(bad_archive), encoding="utf-8")
                    denied, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_tampered_historical_archive_blocks",
                               denied is None and "original context binding" in str(reason))
                    archive_path.write_text(json.dumps(archive_record), encoding="utf-8")
                    ok, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_restored_archive_recovers",
                               ok is not None and reason is None)
                    archive_path.unlink()
                    (archived_model / "model.pt").unlink()
                    denied, reason = verify_linked(linked_migration)
                    check_true("ssot_migrated_context_no_publication_evidence_blocks",
                               denied is None and "original Forward checkpoint SHA publication unavailable" in str(reason))
                    
                    # AI: A new canonical Forward importer captures context and
                    # original FIT execution when provenance is still available.
                    # Later score refreshes must not erase it or require a
                    # Forward copy that no longer exists. Legacy missing-source
                    # checkpoints must continue to fail closed.
                    from services.breakout_quality.fitting_data_dependencies import (
                        build_fitting_context_origin, preserve_fitting_migration_chain,
                    )
                    attested_origin = build_fitting_context_origin(
                        profile=external_context,
                        target=target_lineage,
                        checkpoint=ctx_manifest["artifacts"]["checkpoint"],
                        kind="forward_import",
                        forward_training=source_report["training"],
                        forward_evidence={
                            "training_settings": fitted_settings,
                            "torch_execution": execution,
                            "selected_epoch": ctx_manifest["selected_epoch"],
                        },
                    )
                    attested_fold = {
                        **no_native_epoch,
                        "fitting_origin": attested_origin,
                    }
                    recovered, recovered_reason = verify_linked(
                        attested_fold, checkpoint_to_check=imported_forward_checkpoint,
                    )
                    check_true("ssot_attested_import_survives_forward_source_loss",
                               recovered is not None and recovered_reason is None
                               and recovered.epoch_selection == source_report["training"]["epoch_selection"])
                    tampered_origin = deepcopy(attested_fold)
                    tampered_origin["fitting_origin"]["context_manifest"]["source_stage1"]["selection_score"]["sha256"] = "0" * 64
                    denied, reason = verify_linked(
                        tampered_origin, checkpoint_to_check=imported_forward_checkpoint,
                    )
                    check_true("ssot_attested_import_changed_stage1_blocks",
                               denied is None and "original FIT context witness invalid" in str(reason))
                    tampered_origin = deepcopy(attested_fold)
                    tampered_origin["fitting_origin"]["checkpoint"]["sha256"] = "0" * 64
                    denied, reason = verify_linked(
                        tampered_origin, checkpoint_to_check=imported_forward_checkpoint,
                    )
                    check_true("ssot_attested_import_wrong_checkpoint_sha_blocks",
                               denied is None and "checkpoint SHA binding" in str(reason))
                    tampered_origin = deepcopy(attested_fold)
                    tampered_origin["fitting_origin"]["training_evidence"]["torch_execution"]["deterministic_algorithms"] = False
                    try:
                        verify_linked(tampered_origin, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        execution_blocked = "torch_execution mismatch" in str(exc)
                    else:
                        execution_blocked = False
                    check_true("ssot_attested_import_torch_execution_mismatch_blocks", execution_blocked)
                    tampered_origin = deepcopy(attested_fold)
                    tampered_origin["fitting_origin"]["training_evidence"]["selected_epoch"] += 1
                    try:
                        verify_linked(tampered_origin, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        epoch_blocked = "original fitted epoch differs" in str(exc)
                    else:
                        epoch_blocked = False
                    check_true("ssot_attested_import_wrong_epoch_blocks", epoch_blocked)
                    old_migration = {"kind": "forward_model_fitting_identity_import",
                                     "training_contract_unchanged": True,
                                     "source_checkpoint_sha_preserved": True,
                                     "source_forward_checkpoint": ctx_manifest["artifacts"]["checkpoint"]}
                    first_rescore = preserve_fitting_migration_chain(
                        {"migration": old_migration},
                        {"kind": "checkpoint_score_regeneration",
                         "training_contract_unchanged": True,
                         "source_checkpoint_sha_preserved": True},
                        checkpoint=ctx_manifest["artifacts"]["checkpoint"],
                    )
                    second_rescore = preserve_fitting_migration_chain(
                        {"migration": first_rescore},
                        {"kind": "fitting_identity_checkpoint_reuse",
                         "training_contract_unchanged": True,
                         "source_checkpoint_sha_preserved": True},
                        checkpoint=ctx_manifest["artifacts"]["checkpoint"],
                    )
                    check("ssot_score_only_preserves_earliest_import_source",
                          old_migration, second_rescore["initial_fitting_migration"])
                    try:
                        preserve_fitting_migration_chain(
                            {"migration": old_migration},
                            {"kind": "checkpoint_score_regeneration"},
                            checkpoint={"sha256": "f" * 64, "size_bytes": 10},
                        )
                    except RuntimeError:
                        wrong_source_denied = True
                    else:
                        wrong_source_denied = False
                    check_true("ssot_rescore_cannot_transfer_origin_to_new_checkpoint", wrong_source_denied)
                    from services.breakout_quality.fitted_model_artifacts import (
                        _checkpoint_matches_rescore_contract,
                    )
                    rescore_checkpoint = {
                        **imported_forward_checkpoint,
                        "experiment_profile": external_context,
                        "training_sample_scope": attested_fold["training_sample_scope"],
                        "sequence_length": 300,
                        "feature_count": 10,
                        "context_count": 2,
                    }
                    # Structural sample data only; no inference or weights are fitted.
                    shape_bundle = SimpleNamespace(
                        feature_bank=SimpleNamespace(shape=(1, 300, 10)),
                        group_context=SimpleNamespace(shape=(1, 2)),
                        target_manifest=target_lineage,
                    )
                    check_true("ssot_attested_forward_rescore_twice_without_original_publication",
                               _checkpoint_matches_rescore_contract(
                                   checkpoint=rescore_checkpoint, manifest=attested_fold,
                                   fold_contract=native_context_new, bundle=shape_bundle,
                               ))
                    unproven_rescore = deepcopy(attested_fold)
                    unproven_rescore.pop("fitting_origin")
                    check_true("ssot_legacy_forward_rescore_without_original_proof_blocks",
                               not _checkpoint_matches_rescore_contract(
                                   checkpoint=rescore_checkpoint, manifest=unproven_rescore,
                                   fold_contract=native_context_new, bundle=shape_bundle,
                               ))
                    invalid_training = deepcopy(attested_fold)
                    invalid_training["fitting_origin"]["training_evidence"]["training_settings"]["max_epochs"] += 1
                    check_true("ssot_attested_rescore_wrong_optimizer_training_blocks",
                               not _checkpoint_matches_rescore_contract(
                                   checkpoint=rescore_checkpoint, manifest=invalid_training,
                                   fold_contract=native_context_new, bundle=shape_bundle,
                               ))
                    native_forgery = deepcopy(attested_fold)
                    native_forgery["fitting_origin"]["kind"] = "rolling_native"
                    try:
                        verify_linked(native_forgery, checkpoint_to_check=imported_forward_checkpoint)
                    except RuntimeError as exc:
                        forged_native_blocked = (
                            "original Forward evidence missing" in str(exc)
                            or "lacks embedded native fold contract" in str(exc)
                        )
                    else:
                        forged_native_blocked = False
                    check_true("ssot_forward_import_cannot_claim_rolling_native_lineage",
                               forged_native_blocked)
                    ctx_path.write_text(json.dumps(ctx_manifest), encoding="utf-8")
                    aggregate["folds"][0]["fold_manifest"] = build_file_manifest(ctx_path)
                    aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                    aggregate_path.unlink()
                    denied, reason = owner.validate_rolling_fold_fitted_checkpoint(**ctx_args)
                    check_true("ssot_missing_published_context_binding_fail_closed",
                               denied is None and "aggregate manifest unavailable" in str(reason))

            wrong_execution = owner.validate_rolling_fold_fitted_checkpoint(
                **{**verify_args, "expected_execution": {"device_type": "cuda"}},
            )
            check_true("reverse_bridge_execution_drift_rejected",
                       wrong_execution[0] is None and "torch execution" in str(wrong_execution[1]))
            altered_checkpoint = deepcopy(checkpoint)
            altered_checkpoint["fold_contract"] = deepcopy(historical)
            altered_checkpoint["fold_contract"]["training_settings"]["epochs_max"] = 99
            try:
                owner.validate_rolling_fold_fitted_checkpoint(**{
                    **verify_args, "torch_module": SimpleNamespace(
                        load=lambda *_args, **_kwargs: altered_checkpoint,
                    ),
                })
            except RuntimeError as exc:
                embedded_rejected = "embedded fitting contract mismatch" in str(exc)
            else:
                embedded_rejected = False
            check_true("reverse_bridge_embedded_fitting_drift_fails_closed", embedded_rejected)
            corrupted = dict(manifest)
            corrupted["artifacts"] = {"checkpoint": {"sha256": "bad", "size_bytes": 5}}
            try:
                owner.validate_rolling_fold_fitted_checkpoint(
                    **{**verify_args, "manifest": corrupted},
                )
            except RuntimeError as exc:
                corrupt_rejected = "checkpoint hash mismatch" in str(exc)
            else:
                corrupt_rejected = False
            check_true("reverse_bridge_checkpoint_sha_conflict_fails_closed", corrupt_rejected)
        with patch.object(owner, "source_inventory_matches_current", return_value=False):
            denied, reason = owner.validate_rolling_fold_fitted_checkpoint(**verify_args)
        check_true("reverse_bridge_unproven_full_csv_inventory_fails_closed",
                   denied is None and "source_contract.source_data_inventory mismatch" in str(reason))

    # FIT owner, not Forward mode, handles native/rolling fallback and conflict.
    import json
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

        # If the same-cutoff Rolling Fold exists but cannot supply a complete
        # FIT witness, a missing Forward publication is NOT license to train.
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "missing")), patch.object(
            owner, "resolve_compatible_rolling_first_fold",
            return_value=(None, "UNVERIFIED_ROLLING_FIT: missing original context"),
        ):
            try:
                owner.resolve_forward_fitted_checkpoint(**kwargs)
            except RuntimeError as exc:
                guarded = "automatic retraining" in str(exc)
            else:
                guarded = False
        check_true("cross_mode_forward_ambiguous_fold_never_retrains", guarded)

        # Reverse direction: a published Forward checkpoint with this Fold's
        # cutoff, but missing verified sidecars, must never be silently replaced
        # by Rolling. Different PIT cutoffs and clearly different profiles may
        # legitimately build independent folds.
        fold = {"filter_id": "fixture", "model_architecture": "arch",
                "experiment_profile": "profile", "model_information_cutoff": "2020-12-31"}
        check_true("cross_mode_rolling_potential_first_fold_without_manifest_blocks",
                   owner.unresolved_forward_fit_candidate_issue(
                       model_dir=model_dir, fold_contract=fold,
                       potentially_same_cutoff=True) is not None)
        check("cross_mode_rolling_later_fold_does_not_infer_same_cutoff", None,
              owner.unresolved_forward_fit_candidate_issue(
                  model_dir=model_dir, fold_contract=fold,
                  potentially_same_cutoff=False))
        (model_dir / "manifest.json").write_text(json.dumps({
            **fold, "model_information_cutoff": "2020-12-31",
        }), encoding="utf-8")
        check_true("cross_mode_rolling_matching_forward_requires_proof",
                   owner.unresolved_forward_fit_candidate_issue(
                       model_dir=model_dir, fold_contract=fold) is not None)
        check("cross_mode_rolling_different_cutoff_is_not_same_fit", None,
              owner.unresolved_forward_fit_candidate_issue(
                  model_dir=model_dir, fold_contract={**fold, "model_information_cutoff": "2022-01-01"}))
        check("cross_mode_rolling_other_profile_is_not_same_fit", None,
              owner.unresolved_forward_fit_candidate_issue(
                  model_dir=model_dir, fold_contract={**fold, "experiment_profile": "different"}))

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
                       fold_ids.call_count == 1 and verify.call_count == 1
                       and verify.call_args.kwargs["project_root"] == Path(td))
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

    # Seed-isolated Forward and Rolling jobs must resolve the other mode's
    # published FIT under the SAME seed, never the standard/non-seed root.
    from services.research import breakout_quality_application as cross_mode_app
    with TemporaryDirectory() as td:
        seed_home = Path(td)
        paths = {
            True: (seed_home / "rolling_seed_101", seed_home / "rolling_reports"),
            False: (seed_home / "forward_seed_101", seed_home / "forward_reports"),
        }
        seed_settings = SimpleNamespace(
            filter_id="fixture", model_architecture="fixture_arch",
            experiment_profile="fixture_profile", seed=101,
            point_in_time_resume=True, point_in_time_inner_validation_months=24,
            point_in_time_train_window_months=None,
        )
        def seed_paths(_settings, *, seed, rolling):
            if int(seed) != 101:
                raise AssertionError("wrong benchmark seed")
            return paths[bool(rolling)]
        with patch.object(cross_mode_app, "_model_robustness_dirs", side_effect=seed_paths), patch.object(
            cross_mode_app, "_run_command", return_value=0,
        ) as cli:
            result = cross_mode_app._build_forward_robustness_seed(
                "synthetic", seed_settings, seed=101,
            )
            forward_args = cli.call_args.args[1]
        check_true("cross_mode_forward_robustness_uses_same_seed_rolling_folds",
                   result == 0 and "--rolling-folds-root" in forward_args and
                   forward_args[forward_args.index("--rolling-folds-root") + 1]
                   == str((paths[True][0] / "folds").resolve()))
        with patch.object(cross_mode_app, "_model_robustness_dirs", side_effect=seed_paths), patch.object(
            cross_mode_app, "_rolling_robustness_score_core_ready", return_value=(False, "no score"),
        ), patch.object(cross_mode_app, "_load_forward_robustness_seed", return_value=(None, None, "no report")), patch.object(
            cross_mode_app, "_run_command", return_value=0,
        ) as cli:
            result = cross_mode_app._build_rolling_robustness_seed(
                "synthetic", seed_settings, seed=101, inputs_prepared=True,
            )
            rolling_args = cli.call_args_list[0].args[1]
        check_true("cross_mode_rolling_robustness_inspects_same_seed_forward_fit",
                   result == 0 and "--forward-fit-candidate-model-dir" in rolling_args and
                   rolling_args[rolling_args.index("--forward-fit-candidate-model-dir") + 1]
                   == str(paths[False][0]))

    # Standard Rolling uses a PIT directory override; it must still locate the
    # matching canonical Forward candidate before permitting a new optimizer.
    with TemporaryDirectory() as td:
        standard_home = Path(td)
        standard_settings = SimpleNamespace(
            filter_id="fixture", model_architecture="fixture_arch",
            experiment_profile="fixture_profile", seed=101,
            point_in_time_resume=True, point_in_time_inner_validation_months=24,
            point_in_time_train_window_months=None,
        )
        standard_mode = SimpleNamespace(
            score_start_date="2021-01-01", score_end_date=None,
            fold_months=12, fold_anchor_date=None, single_score_block=False,
        )
        candidate_dir = standard_home / "forward_fit"
        pit_dir = standard_home / "rolling_pit_override"
        with patch.object(cross_mode_app, "get_breakout_quality_workflow_settings",
                          return_value=standard_settings), patch.object(
            cross_mode_app, "_rolling_mode_point_in_time_dir", return_value=pit_dir,
        ), patch.object(cross_mode_app, "_forward_publication_paths",
                        return_value=(candidate_dir, standard_home / "forward_report")), patch.object(
            cross_mode_app, "_prepare_continuous_research_inputs", return_value=0,
        ), patch.object(cross_mode_app, "_run_command", return_value=1) as cli:
            # Producer failure is deliberate: inspect the forwarded CLI flags
            # without changing any real training / score / FIT artifacts.
            result = cross_mode_app._run_continuous_pit_profile(
                "synthetic", model_id="fixture", profile_name="fixture_profile",
                mode=standard_mode,
            )
            standard_args = cli.call_args.args[1]
        check_true("cross_mode_standard_rolling_override_inspects_forward_fit",
                   result == 1 and "--point-in-time-dir-override" in standard_args
                   and "--forward-fit-candidate-model-dir" in standard_args
                   and standard_args[standard_args.index("--forward-fit-candidate-model-dir") + 1]
                   == str(candidate_dir) and "--resume" in standard_args)

    # Snapshot caches must observe in-memory fault-injection source changes.
    # Their keys cannot be only a module filename, or tests would miss a
    # newly introduced forbidden readiness helper with the same path.
    from .research_lifecycle_architecture import (
        _parse_snapshot_source, _top_level_function_facts,
    )
    cache_file = "synthetic/research_snapshot_cache_guard.py"
    healthy_source = "def probe(path):\n    return path.name\n"
    mutated_source = "def probe(path):\n    return (path / 'model.pt').exists()\n"
    check_true("snapshot_ast_cache_reparses_mutated_source",
               _parse_snapshot_source(cache_file, healthy_source)
               is not _parse_snapshot_source(cache_file, mutated_source))
    healthy_facts = _top_level_function_facts(cache_file, healthy_source)["probe"]
    mutated_facts = _top_level_function_facts(cache_file, mutated_source)["probe"]
    check_true("snapshot_function_fact_cache_recomputes_mutated_source",
               "model.pt" not in healthy_facts.tokens
               and "model.pt" in mutated_facts.tokens
               and not healthy_facts.path_probe and mutated_facts.path_probe)

    # Explicit provenance repair has a different responsibility from FIT REUSE.
    # Test the transaction with real directory operations and synthetic model
    # bytes, without invoking training or modifying current data/artifacts.
    from services.breakout_quality.fit_repair_transaction import (
        preserved_fit_publications, recover_incomplete_fit_repair,
    )
    from hashlib import sha256 as _sha256
    with TemporaryDirectory() as td:
        root = Path(td)
        mod = root / "models" / "model_profile"
        out = root / "outputs" / "model_profile"
        mod.mkdir(parents=True)
        out.mkdir(parents=True)
        (mod / "model.pt").write_bytes(b"original_model")
        (out / "report.json").write_bytes(b"original_report")
        journal = root / "audit" / "fitting.in_progress.json"
        with preserved_fit_publications(
            project_root=root, publication_roots=(mod, out, mod / "point_in_time"),
            journal_path=journal, repair_identity="fixture",
        ):
            mod.mkdir(parents=True)
            out.mkdir(parents=True)
            (mod / "model.pt").write_bytes(b"new_model")
            (out / "report.json").write_bytes(b"new_report")
            check_true("fit_repair_old_publication_is_staged_before_canonical_build",
                       not any(mod.parent.glob("historical_fit_repairs/model_profile/*/original/model.pt")))
        archived = sorted(mod.parent.glob("historical_fit_repairs/model_profile/*/original/model.pt"))
        check_true("fit_repair_preserves_old_model_and_output_on_commit",
                   len(archived) == 1 and archived[0].read_bytes() == b"original_model"
                   and (mod / "model.pt").read_bytes() == b"new_model"
                   and len(list(out.parent.glob("historical_fit_repairs/model_profile/*/original/report.json"))) == 1
                   and not journal.exists())
        try:
            with preserved_fit_publications(
                project_root=root, publication_roots=(mod, out),
                journal_path=journal, repair_identity="fixture",
            ):
                mod.mkdir(parents=True)
                (mod / "model.pt").write_bytes(b"invalid_new_model")
                raise RuntimeError("simulate producer failure")
        except RuntimeError as exc:
            check_true("fit_repair_producer_failure_is_visible", "simulate producer failure" in str(exc))
        check_true("fit_repair_rolls_back_both_publications_on_failure",
                   (mod / "model.pt").read_bytes() == b"new_model"
                   and (out / "report.json").read_bytes() == b"new_report"
                   and not journal.exists())
        recovery = root / "audit" / "interrupted.in_progress.json"
        held = mod.with_name(".model_profile.fit_repair_pending_fixture")
        mod.rename(held)
        recovery.write_text(json.dumps({
            "schema": "fit_repair_transaction_v1", "state": "in_progress",
            "sources": [{"source": str(mod), "hold": str(held),
                         "archive": str(mod.parent / "historical_fit_repairs" / mod.name / "fixture" / "original"),
                         "existed": True}],
        }), encoding="utf-8")
        mod.mkdir()
        (mod / "model.pt").write_bytes(b"incomplete_model")
        recover_incomplete_fit_repair(journal_path=recovery)
        check_true("fit_repair_recovers_after_interrupted_publication",
                   (mod / "model.pt").read_bytes() == b"new_model"
                   and not recovery.exists())
        try:
            with preserved_fit_publications(
                project_root=root, publication_roots=(root.parent, ),
                journal_path=journal, repair_identity="fixture",
            ):
                pass
        except RuntimeError:
            outside_blocked = True
        else:
            outside_blocked = False
        check_true("fit_repair_rejects_outside_project_publications", outside_blocked)

    # FIT authority checks *real bytes* and producer-bound native origins,
    # not model names, READY status or reports reconstructed from later context.
    from services.breakout_quality.fitted_model_artifacts import (
        verify_fit_repair_previous_checkpoint,
        verify_fit_repair_rebuilt_checkpoint,
        verified_fit_repair_legacy_cache_entry, fold_training_identity,
    )
    from filters.breakout_quality.artifacts import build_file_manifest
    import json
    with TemporaryDirectory() as td:
        root = Path(td)
        fold = root / "rolling" / "folds" / "fold_20210101_20211231"
        fold.mkdir(parents=True)
        model = fold / "model.pt"
        model.write_bytes(b"verified canonical Rolling FIT bytes")
        model_record = build_file_manifest(model)
        historical_manifest = {
            "experiment_profile": "profile_fixture",
            "planned_periods": {"score_start": "2021-01-01"},
            "artifacts": {"checkpoint": model_record},
        }
        (fold / "manifest.json").write_text(json.dumps(historical_manifest))
        sha = str(model_record["sha256"])
        verify_fit_repair_previous_checkpoint(
            rolling_dir=root / "rolling", score_start="2021-01-01",
            expected_profile="profile_fixture", expected_checkpoint_sha256=sha,
        )
        check_true("fit_repair_old_checkpoint_sha_requires_exact_published_bytes",
                   len(sha) == 64)
        # The stale shared cache is not a new origin witness; quarantine only
        # the exact first-Fold FitKey and require byte/manifest equality.
        cache_root = root / "shared_fitting_checkpoints"
        cache_entry = cache_root / fold_training_identity(historical_manifest)
        cache_entry.mkdir(parents=True)
        (cache_entry / "model.pt").write_bytes(model.read_bytes())
        (cache_entry / "manifest.json").write_text(json.dumps(historical_manifest))
        cache_selected = verified_fit_repair_legacy_cache_entry(
            rolling_dir=root / "rolling", score_start="2021-01-01",
            expected_profile="profile_fixture", checkpoint_cache_root=cache_root,
        )
        check_true("fit_repair_locates_only_same_fitkey_shared_cache_entry",
                   cache_selected == cache_entry)
        (cache_entry / "model.pt").write_bytes(b"conflicting cached optimizer")
        try:
            verified_fit_repair_legacy_cache_entry(
                rolling_dir=root / "rolling", score_start="2021-01-01",
                expected_profile="profile_fixture", checkpoint_cache_root=cache_root,
            )
        except RuntimeError:
            conflicting_cache_blocked = True
        else:
            conflicting_cache_blocked = False
        check_true("fit_repair_blocks_same_fitkey_different_shared_cache_weights",
                   conflicting_cache_blocked)
        (cache_entry / "model.pt").write_bytes(model.read_bytes())
        def old_fit_rejected(**overrides):
            call = dict(rolling_dir=root / "rolling", score_start="2021-01-01",
                        expected_profile="profile_fixture", expected_checkpoint_sha256=sha)
            call.update(overrides)
            try:
                verify_fit_repair_previous_checkpoint(**call)
            except RuntimeError:
                return True
            return False
        check_true("fit_repair_rejects_changed_preflight_checkpoint_sha",
                   old_fit_rejected(expected_checkpoint_sha256="b" * 64))
        native = {**historical_manifest,
                  "fitting_origin": {"kind": "rolling_native", "checkpoint": model_record}}
        (fold / "manifest.json").write_text(json.dumps(native))
        forwarded = root / "forward" / "model.pt"
        forwarded.parent.mkdir()
        forwarded.write_bytes(model.read_bytes())
        rebuilt = verify_fit_repair_rebuilt_checkpoint(
            rolling_dir=root / "rolling", score_start="2021-01-01",
            expected_profile="profile_fixture", forward_checkpoint=forwarded,
        )
        check_true("fit_repair_new_forward_has_same_native_rolling_fit_sha", rebuilt == sha)
        forwarded.write_bytes(b"another optimizer unexpectedly retrained Forward")
        try:
            verify_fit_repair_rebuilt_checkpoint(
                rolling_dir=root / "rolling", score_start="2021-01-01",
                expected_profile="profile_fixture", forward_checkpoint=forwarded,
            )
        except RuntimeError:
            second_optimizer_blocked = True
        else:
            second_optimizer_blocked = False
        check_true("fit_repair_rejects_independent_forward_optimizer_checkpoint",
                   second_optimizer_blocked)
        forwarded.write_bytes(model.read_bytes())
        native["fitting_origin"] = {"kind": "forward_import", "checkpoint": model_record}
        (fold / "manifest.json").write_text(json.dumps(native))
        try:
            verify_fit_repair_rebuilt_checkpoint(
                rolling_dir=root / "rolling", score_start="2021-01-01",
                expected_profile="profile_fixture", forward_checkpoint=forwarded,
            )
        except RuntimeError:
            imported_not_native = True
        else:
            imported_not_native = False
        check_true("fit_repair_rejects_old_import_disguised_as_native_rebuild",
                   imported_not_native)

    # If the archive step committed before a process interruption, recovery
    # must only clear its journal, never revert a proven new publication.
    with TemporaryDirectory() as td:
        root = Path(td)
        published = root / "models" / "p"
        published.mkdir(parents=True)
        (published / "model.pt").write_bytes(b"committed new model")
        archived = root / "models" / "historical_fit_repairs" / "p" / "stamp" / "original"
        archived.mkdir(parents=True)
        (archived / "model.pt").write_bytes(b"old model")
        pending = published.with_name(".p.fit_repair_pending_stamp")
        journal = root / "audit" / "committed.in_progress.json"
        journal.parent.mkdir()
        journal.write_text(json.dumps({"schema": "fit_repair_transaction_v1", "state": "committed",
            "sources": [{"source": str(published), "hold": str(pending),
                         "archive": str(archived), "existed": True}]}))
        recover_incomplete_fit_repair(journal_path=journal)
        check_true("fit_repair_committed_journal_recovery_keeps_new_fitted_truth",
                   (published / "model.pt").read_bytes() == b"committed new model"
                   and (archived / "model.pt").read_bytes() == b"old model"
                   and not journal.exists())

    # Exercise the actual Research orchestrator with byte-backed FIT fixtures.
    # Forward comparison must stage originals and call ONE canonical Forward
    # trainer. No Rolling fold rebuild is permitted for a Forward-only repair.
    from contextlib import ExitStack
    from services.research import breakout_quality_application as app
    for invalid_forward in (False, True):
        with TemporaryDirectory() as td:
            root = Path(td)
            mod, out = root / "models" / "profile", root / "outputs" / "profile"
            rolling = mod / "point_in_time"
            fold = rolling / "folds" / "fold_20210101_20211231"
            fold.mkdir(parents=True)
            out.mkdir(parents=True)
            old_checkpoint = fold / "model.pt"
            old_checkpoint.write_bytes(b"old unverified source checkpoint")
            old_record = build_file_manifest(old_checkpoint)
            (fold / "manifest.json").write_text(json.dumps({
                "experiment_profile": "profile_fixture",
                "planned_periods": {"score_start": "2021-01-01"},
                "artifacts": {"checkpoint": old_record},
            }))
            (mod / "model.pt").write_bytes(b"historical Forward checkpoint")
            (out / "report.json").write_text("historical report")
            from config.breakout_quality import BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT
            legacy_cache = (root / BREAKOUT_QUALITY_SHARED_FITTING_CHECKPOINT_CACHE_ROOT
                            / fold_training_identity(json.loads((fold / "manifest.json").read_text())))
            legacy_cache.mkdir(parents=True)
            (legacy_cache / "model.pt").write_bytes(old_checkpoint.read_bytes())
            (legacy_cache / "manifest.json").write_bytes((fold / "manifest.json").read_bytes())
            settings = SimpleNamespace(
                rolling_authorized=True, filter_id="fixture_filter",
                model_architecture="fixture_arch", experiment_profile="profile_fixture",
            )
            stages = []
            def canonical_rolling(*_args, **_kwargs):
                raise AssertionError("Forward comparison must not train Rolling folds")
            def forward_evaluation(*_args, **kwargs):
                check_true("forward_repair_builds_only_one_fresh_forward_fit_"
                           + str(invalid_forward),
                           kwargs["force_rebuild"] is True
                           and kwargs["prompt_for_build"] is False
                           and "require_rolling_fit_reuse" not in kwargs
                           and not (mod / "model.pt").exists()
                           and not rolling.exists()
                           and not legacy_cache.exists())
                mod.mkdir(parents=True, exist_ok=True)
                (mod / "model.pt").write_bytes(b"one newly trained Forward FIT")
                (mod / "manifest.json").write_text(json.dumps({
                    "model": build_file_manifest(mod / "model.pt"), "selected_epoch": 1,
                }), encoding="utf-8")
                out.mkdir(parents=True, exist_ok=True)
                (out / "report.json").write_text("new Forward report")
                stages.append("forward")
                return 0, object(), "REUSE" if invalid_forward else "FORCE_REBUILD"
            mock_values = {
                "PROJECT_ROOT": root,
                "get_breakout_quality_model_test_settings": lambda: SimpleNamespace(
                    model_profiles=(("MR-FIXTURE", "profile_fixture"),)),
                "get_breakout_quality_workflow_settings": lambda **_kw: settings,
                "get_breakout_quality_rolling_test_mode": lambda _mode: SimpleNamespace(
                    score_start_date="2021-01-01"),
                "_model_display_id": lambda _profile: "MR-FIXTURE",
                "_forward_publication_paths": lambda _settings: (mod, out),
                "_rolling_publication_dir": lambda _settings, _mode: rolling,
                "_collect_continuous_research_input_plan": lambda _settings: SimpleNamespace(
                    blocked=False, actions=()),
                "_run_continuous_pit_profile": canonical_rolling,
                "_load_reusable_rolling_standard_report": lambda _settings, _mode: (object(), {}, None),
                "invalidate_rolling_model_source_cache": lambda: None,
                "_clear_continuous_forward_reuse_caches": lambda: None,
                "_ensure_continuous_forward_model_report": forward_evaluation,
                "resolve_filter_artifact_paths": lambda *_args, **_kw: SimpleNamespace(
                    model_path=mod / "model.pt", manifest_path=mod / "manifest.json"),
                "_load_reusable_continuous_forward_contract": lambda _settings: (object(), ""),
            }
            with ExitStack() as stack:
                for attr, value in mock_values.items():
                    stack.enter_context(patch.object(app, attr, value))
                try:
                    result = app.repair_missing_fit_origin_using_forward(
                        "apps/research.py", profile_name="profile_fixture",
                        expected_previous_checkpoint_sha256=str(old_record["sha256"]),
                    )
                except RuntimeError:
                    failure = True
                    result = None
                else:
                    failure = False
            if invalid_forward:
                check_true("repair_rolls_back_after_incompatible_forward_fit",
                           failure and stages == ["forward"]
                           and (fold / "model.pt").read_bytes() == b"old unverified source checkpoint"
                           and (mod / "model.pt").read_bytes() == b"historical Forward checkpoint"
                           and (out / "report.json").read_text() == "historical report"
                           and (legacy_cache / "model.pt").read_bytes()
                           == b"old unverified source checkpoint")
            else:
                preserved = sorted(mod.parent.glob(
                    "historical_fit_repairs/profile/*/original/point_in_time/folds/*/model.pt"))
                stale_cache_archives = sorted(legacy_cache.parent.glob(
                    "historical_fit_repairs/" + legacy_cache.name + "/*/original/model.pt"))
                check_true("repair_commits_one_forward_fit_and_preserves_old_rolling_history",
                           not failure and stages == ["forward"]
                           and result is not None
                           and result["repaired_checkpoint_sha256"]
                           == build_file_manifest(mod / "model.pt")["sha256"]
                           and len(preserved) == 1
                           and preserved[0].read_bytes() == b"old unverified source checkpoint"
                           and not legacy_cache.exists()
                           and not rolling.exists()  # old six-fold OOS is historical, not READY
                           and len(stale_cache_archives) == 1
                           and stale_cache_archives[0].read_bytes()
                           == b"old unverified source checkpoint")

    # New Forward evidence is verified by the FIT owner, not by a filename
    # probe in the Research orchestration layer.
    from services.breakout_quality.fitted_model_artifacts import (
        verified_new_forward_fit_repair_checkpoint,
    )
    with TemporaryDirectory() as td:
        d = Path(td)
        model_path, manifest_path = d / "model.pt", d / "manifest.json"
        model_path.write_bytes(b"new native Forward checkpoint")
        current_record = build_file_manifest(model_path)
        manifest_path.write_text(json.dumps({
            "selected_epoch": 1, "model": current_record,
        }), encoding="utf-8")
        check_true("new_forward_fit_repair_uses_canonical_manifest_checkpoint_sha",
                   verified_new_forward_fit_repair_checkpoint(
                       model_path=model_path, manifest_path=manifest_path,
                   ) == current_record["sha256"])
        model_path.write_bytes(b"tampered model weights")
        rejected = False
        try:
            verified_new_forward_fit_repair_checkpoint(
                model_path=model_path, manifest_path=manifest_path,
            )
        except RuntimeError:
            rejected = True
        check_true("new_forward_fit_repair_rejects_corrupted_checkpoint_after_training", rejected)
        model_path.write_bytes(b"new native Forward checkpoint")
        manifest_path.write_text(json.dumps({
            "selected_epoch": 0, "model": current_record,
        }), encoding="utf-8")
        rejected = False
        try:
            verified_new_forward_fit_repair_checkpoint(
                model_path=model_path, manifest_path=manifest_path,
            )
        except RuntimeError:
            rejected = True
        check_true("new_forward_fit_repair_rejects_missing_epoch_evidence", rejected)

    # A Ctrl+C/terminated process must not create a manual --recover prerequisite.
    from services.breakout_quality.fit_repair_transaction import (
        fit_repair_journal_path, _fit_repair_lock,
    )
    with TemporaryDirectory() as td:
        root = Path(td)
        from services.research import breakout_quality_application as app
        model_root = root / "models" / "fixture_profile"
        model_root.mkdir(parents=True)
        (model_root / "model.pt").write_bytes(b"new uncommitted FIT")
        pending = model_root.with_name(".fixture_profile.fit_repair_pending_aborted")
        pending.mkdir()
        (pending / "model.pt").write_bytes(b"original FIT")
        journal = fit_repair_journal_path(
            project_root=root, filter_id="test_filter", profile="fixture_profile",
        )
        journal.parent.mkdir(parents=True)
        journal.write_text(json.dumps({
            "schema": "fit_repair_transaction_v1", "state": "in_progress",
            "sources": [{"source": str(model_root), "hold": str(pending),
                         "archive": str(model_root.parent / "historical_fit_repairs"
                                        / model_root.name / "aborted" / "original"),
                         "existed": True}],
        }), encoding="utf-8")
        test_settings = SimpleNamespace(filter_id="test_filter")
        with patch.object(app, "PROJECT_ROOT", root), \
             patch.object(app, "get_breakout_quality_workflow_settings", return_value=test_settings), \
             patch.object(app, "invalidate_rolling_model_source_cache", lambda: None), \
             patch.object(app, "_clear_continuous_forward_reuse_caches", lambda: None):
            count = app._recover_pending_model_fit_repairs((("TEST", "fixture_profile"),))
        check_true("normal_research_auto_recovers_abandoned_fit_publications",
                   count == 1 and not journal.exists() and not pending.exists()
                   and (model_root / "model.pt").read_bytes() == b"original FIT")
        empty_journal = journal
        empty_journal.touch()
        outcome = recover_incomplete_fit_repair(journal_path=empty_journal, project_root=root)
        check_true("normal_research_clears_interrupted_empty_transaction_marker",
                   outcome == "EMPTY_JOURNAL_CLEARED" and not empty_journal.exists())
        # A second process may not reclaim a live staged FIT even if the
        # journal exists. The same cross-platform kernel lock guards both.
        journal.write_text("{}", encoding="utf-8")
        lock_blocked = False
        with _fit_repair_lock(journal):
            try:
                recover_incomplete_fit_repair(journal_path=journal, project_root=root)
            except RuntimeError as exc:
                lock_blocked = "still active" in str(exc)
        check_true("fit_recovery_does_not_touch_live_repair_transaction",
                   lock_blocked and journal.read_text(encoding="utf-8") == "{}")

    # Shared FIT-origin preflight is a formal Research dependency capability,
    # not a manual audit command or an MR identity switch.
    from core.breakout_quality_policy import (
        get_breakout_quality_model_test_settings as configured_models,
        get_breakout_quality_workflow_settings as configured_settings,
        get_breakout_quality_rolling_test_mode as configured_rolling_mode,
    )
    from services.breakout_quality.fit_origin_repair_plan import (
        FitOriginRepairCandidate, FitOriginRepairBlocker, plan_fit_origin_repairs,
    )
    from services.breakout_quality.fitting_data_dependencies import context_source_for_profile
    from filters.breakout_quality.predicted_context_artifact import maybe_predicted_context_artifact_spec
    from services.breakout_quality.fitted_model_artifacts import resolve_rolling_fitted_folds_root
    from filters.breakout_quality.paths import (
        SELECTION_POINT_IN_TIME_MANIFEST_FILENAME, resolve_filter_artifact_paths,
    )
    from services.research import breakout_quality_application as app

    external_profiles = [
        (str(model_id), str(profile)) for model_id, profile in configured_models().model_profiles
        if maybe_predicted_context_artifact_spec(context_source_for_profile(str(profile))) is not None
    ]
    check_true("normal_research_repair_profile_registry_driven", bool(external_profiles))
    if external_profiles:
        model_id, profile = external_profiles[0]
        settings = configured_settings(experiment_profile=profile)
        score_start = str(configured_rolling_mode("rolling").score_start_date)
        with TemporaryDirectory() as td:
            project = Path(td)
            fold_root = resolve_rolling_fitted_folds_root(
                project_root=project, filter_id=str(settings.filter_id),
                model_architecture=str(settings.model_architecture), experiment_profile=profile,
            )
            fold = fold_root / "fold_source_fixture"
            fold.mkdir(parents=True)
            (fold / "model.pt").write_bytes(b"unverified source for native FIT rebuild")
            original_record = build_file_manifest(fold / "model.pt")
            legacy = {
                "fold_id": "fold_source_fixture", "experiment_profile": profile,
                "planned_periods": {"score_start": score_start},
                "artifacts": {"checkpoint": original_record},
                "migration": {
                    "kind": "expanded_daily_score_universe_checkpoint_reuse",
                    "source_checkpoint_sha_preserved": True,
                    "training_contract_unchanged": True,
                },
            }
            fold_manifest = fold / "manifest.json"
            aggregate_path = fold_root.parent / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
            def write_case():
                fold_manifest.write_text(json.dumps(legacy), encoding="utf-8")
                aggregate_path.write_text(json.dumps({"folds": [{
                    "fold_id": "fold_source_fixture",
                    "fold_manifest": build_file_manifest(fold_manifest),
                }]}), encoding="utf-8")
            write_case()
            kwargs = {"project_root": project, "model_profiles": ((model_id, profile),)}
            jobs, blockers = plan_fit_origin_repairs(**kwargs)
            check_true("normal_research_detects_missing_origin_in_registry_context",
                       len(jobs) == 1 and not blockers
                       and jobs[0].previous_checkpoint_sha256 == original_record["sha256"])
            paths = resolve_filter_artifact_paths(
                project, str(settings.filter_id), str(settings.model_architecture), profile,
            )
            paths.model_dir.mkdir(parents=True, exist_ok=True)
            paths.model_path.write_bytes((fold / "model.pt").read_bytes())
            jobs, blockers = plan_fit_origin_repairs(**kwargs)
            check_true("normal_research_never_rebuilds_with_matching_forward_witness",
                       not jobs and not blockers)
            paths.model_path.write_bytes(b"other published Forward weight")
            legacy["fitting_origin"] = {"kind": "rolling_native"}
            write_case()
            jobs, blockers = plan_fit_origin_repairs(**kwargs)
            check_true("normal_research_never_relabels_existing_fitting_origin",
                       not jobs and not blockers)
            legacy.pop("fitting_origin")
            legacy["artifacts"]["checkpoint"]["sha256"] = "0" * 64
            write_case()
            jobs, blockers = plan_fit_origin_repairs(**kwargs)
            check_true("normal_research_refuses_corrupt_checkpoint_in_all_model_preflight",
                       not jobs and len(blockers) == 1
                       and "SHA mismatch" in blockers[0].reason)
            legacy["artifacts"]["checkpoint"] = original_record
            write_case()
            aggregate_path.unlink()
            jobs, blockers = plan_fit_origin_repairs(**kwargs)
            check_true("normal_research_refuses_missing_aggregate_binding",
                       not jobs and len(blockers) == 1)

        # A comparison is an evaluation consumer, never permission to replace
        # existing historical weights because their original attestation is stale.
        # Complete FIT evidence may be reused; unproven lineage must BLOCK before
        # any publication mutation or training, regardless of profile identity.
        for case in ("candidate", "block", "ready"):
            with TemporaryDirectory() as td:
                project = Path(td)
                statuses = []
                report_written = []
                repaired = []
                state = {"ready": case == "ready"}
                scenario_settings = SimpleNamespace(experiment_profile=profile)
                candidate = FitOriginRepairCandidate(model_id, profile, "a" * 64)
                def readiness(*_args, **_kwargs):
                    ok = state["ready"]
                    return SimpleNamespace(
                        source_contract=object() if ok else None,
                        evidence_payload={"model_research_id": model_id} if ok else None,
                        display_reason="READY" if ok else "FIT lineage gap",
                        display_action="REUSE" if ok else "VERIFY FIT",
                        evidence_ready=ok,
                    )
                def repair(_program, *, profile_name, expected_previous_checkpoint_sha256):
                    repaired.append(profile_name)
                    if case == "rollback":
                        raise RuntimeError("synthetic canonical rebuild failure; old FIT retained")
                    state["ready"] = True
                    return {"repaired_checkpoint_sha256": "b" * 64}
                mock_map = {
                    "PROJECT_ROOT": project,
                    "get_breakout_quality_model_test_settings": lambda: SimpleNamespace(
                        model_profiles=((model_id, profile),)),
                    "get_breakout_quality_workflow_settings": lambda **_kw: scenario_settings,
                    "_resolve_continuous_model_readiness": readiness,
                    "_collect_continuous_research_input_plan": lambda _settings: SimpleNamespace(blocked=False),
                    "plan_fit_origin_repairs": lambda **_kw: (
                        (candidate,) if case == "candidate" else (),
                        (FitOriginRepairBlocker(model_id, profile, "synthetic invalid source"),)
                        if case == "block" else (),
                    ),
                    "repair_missing_fit_origin_using_forward": repair,
                    "_recover_pending_model_fit_repairs": lambda _models: 0,
                    "invalidate_rolling_model_source_cache": lambda: None,
                    "_clear_continuous_forward_reuse_caches": lambda: None,
                    "_print_model_action_status": lambda rows: statuses.extend(rows),
                    "_render_standard_model_comparison": lambda *_a, **_kw: "comparison",
                    "_write_standard_model_comparison_report": lambda *_a, **_kw: (
                        report_written.append(True) or project / "result.json", project / "result.md"),
                    "render_status_paths": lambda *_a, **_kw: "artifact paths",
                    "render_title": lambda label: label,
                }
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for attr, value in mock_map.items():
                        stack.enter_context(patch.object(app, attr, value))
                    result_code = app._run_configured_model_comparison(
                        "apps/research.py model", rolling=False,
                    )
                if case == "candidate":
                    check_true("normal_research_unproven_fit_blocks_without_refitting",
                               result_code == 2 and not repaired
                               and not report_written and statuses[0][2] == "BLOCKED")
                elif case == "block":
                    check_true("normal_research_preflight_blocks_before_all_mutations",
                               result_code == 2 and not repaired and not report_written)
                else:
                    check_true("normal_research_verified_fit_reuses_without_rebuilding",
                               result_code == 0 and not repaired
                               and bool(report_written) and statuses[0][2] == "REUSE")

    # Rolling report refresh is not a FIT-origin defect: the shared entry must
    # not start a GPU rebuild when its published SCORE is already valid.
    if external_profiles:
        model_id, profile = external_profiles[0]
        with TemporaryDirectory() as td:
            project = Path(td)
            report_refresh_calls = []
            plan_inputs = []
            settings = SimpleNamespace(experiment_profile=profile, filter_id="synthetic")
            contract = SimpleNamespace(audit_path=str(project / "old_audit.json"))
            def only_report_stale(*_a, **_kw):
                return SimpleNamespace(
                    source_contract=contract, evidence_payload=None,
                    display_reason="Rolling Audit refresh required",
                    display_action="BUILD/REFRESH", evidence_ready=False,
                )
            def record_plan(**kw):
                plan_inputs.append(tuple(kw["model_profiles"]))
                return (), ()
            mocked = {
                "PROJECT_ROOT": project,
                "get_breakout_quality_model_test_settings": lambda: SimpleNamespace(
                    model_profiles=((model_id, profile),)),
                "get_breakout_quality_workflow_settings": lambda **_kw: settings,
                "_resolve_continuous_model_readiness": only_report_stale,
                "_load_canonical_forward_checkpoint_import_source": lambda _settings: (None, None),
                "_collect_continuous_research_input_plan": lambda _settings: SimpleNamespace(blocked=False),
                "plan_fit_origin_repairs": record_plan,
                "_run_continuous_rolling_mode_direct": lambda *_a, **_kw: (
                    report_refresh_calls.append("audit_refresh") or 0),
                "invalidate_rolling_model_source_cache": lambda: None,
                "_load_reusable_rolling_standard_report": lambda *_a, **_kw: (contract, {"model_research_id": model_id}, None),
                "_print_model_action_status": lambda *_a: None,
                "_render_standard_model_comparison": lambda *_a, **_kw: "comparison",
                "_write_standard_model_comparison_report": lambda *_a, **_kw: (
                    project / "model.json", project / "model.md"),
                "render_status_paths": lambda *_a, **_kw: "artifact paths",
                "render_title": lambda label: label,
            }
            from contextlib import ExitStack
            with ExitStack() as stack:
                for key, value in mocked.items():
                    stack.enter_context(patch.object(app, key, value))
                code = app._run_configured_model_comparison("apps/research.py model", rolling=True)
            check_true("rolling_audit_only_refresh_never_rebuilds_fitting_origin",
                       code == 0 and report_refresh_calls == ["audit_refresh"]
                       and plan_inputs == [()])

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
        "    seed_states, verified_sources = _observe_model_robustness_sources(",
        "    ghost_score = object()\n"
        "    seed_states, verified_sources = _observe_model_robustness_sources(", 1,
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
    edited[MODEL_SCORE_OWNER] = edited[MODEL_SCORE_OWNER].replace(
        "    score_source = load_selection_point_in_time_score_source_contract(",
        "    score_source = bypassed_score_validation(", 1,
    )
    check_true("delegated_complete_pit_ranking_must_reach_canonical_score_owner",
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
        score.write_text(
            "ticker,date,group_index,breakout_quality_score,fold_id,model_information_cutoff\n"
            "0050,2021-01-04,1,0.7,fold_20210104_20211231,2020-12-31\n",
            encoding="utf-8",
        )
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
            # A PIT bundle can stay byte-identical while a different process
            # publishes a new upstream Dataset. Revalidation must not depend
            # solely on a PIT Score/Audit producer calling cache_clear().
            from filters.breakout_quality.dataset_store import resolve_dataset_paths
            from filters.breakout_quality.paths import resolve_filter_output_dir
            dataset_summary_path = resolve_dataset_paths(
                resolve_filter_output_dir(directory, str(settings.filter_id))
            ).summary
            dataset_summary_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_summary_path.write_text('{"revision": "a"}', encoding="utf-8")
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
            score_owner.load_selection_point_in_time_ranking_contract(**kwargs)
            original_dataset_stat = dataset_summary_path.stat()
            staged_dataset = dataset_summary_path.with_name(".dataset_next.json")
            staged_dataset.write_text('{"revision": "b"}', encoding="utf-8")
            import os
            os.utime(staged_dataset, ns=(
                original_dataset_stat.st_atime_ns, original_dataset_stat.st_mtime_ns,
            ))
            os.replace(staged_dataset, dataset_summary_path)
            with patch.object(
                score_owner, "validate_current_pit_score_source_lineage",
                side_effect=ValueError("PIT score Dataset/Target provenance stale"),
            ) as revalidated_upstream:
                try:
                    score_owner.load_selection_point_in_time_ranking_contract(**kwargs)
                except ValueError as exc:
                    changed_dataset_rejected = "provenance stale" in str(exc)
                else:
                    changed_dataset_rejected = False
            current_dataset_stat = dataset_summary_path.stat()
            check_true("pit_cached_ranking_rechecks_republished_dataset_without_pit_change",
                       changed_dataset_rejected and revalidated_upstream.call_count == 1
                       and original_dataset_stat.st_size == current_dataset_stat.st_size
                       and original_dataset_stat.st_mtime_ns == current_dataset_stat.st_mtime_ns)
            dataset_summary_path.unlink()
            score_owner.clear_selection_point_in_time_ranking_contract_cache()

            # Event-target profiles have a separate canonical Target manifest.
            # A new Target publication must likewise invalidate the cached PIT
            # Ranking contract even if all four PIT publication files are frozen.
            from core.breakout_quality_registry import (
                CONTINUOUS_RANKER_TRAINING_OBJECTIVES,
                SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES,
            )
            from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
            from filters.breakout_quality.continuous_target import (
                TARGET_MANIFEST_FILENAME, resolve_continuous_target_dir,
            )
            event_profile = next(
                name for name in SUPPORTED_BREAKOUT_QUALITY_EXPERIMENT_PROFILES
                if get_breakout_quality_experiment_profile(name).training_objective
                in CONTINUOUS_RANKER_TRAINING_OBJECTIVES
                if get_continuous_ranker_execution_recipe(name).dependency_spec.requires_continuous_target_artifact
            )
            event_target = get_breakout_quality_experiment_profile(event_profile).continuous_target_id
            target_path = resolve_continuous_target_dir(
                directory, str(settings.filter_id), target_id=str(event_target),
            ) / TARGET_MANIFEST_FILENAME
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text('{"revision": "a"}', encoding="utf-8")
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
            event_kwargs = {**kwargs, "experiment_profile": event_profile}
            with patch.object(
                score_owner, "_load_selection_point_in_time_ranking_contract_uncached",
                return_value=complete,
            ) as canonical_gate:
                score_owner.load_selection_point_in_time_ranking_contract(**event_kwargs)
                score_owner.load_selection_point_in_time_ranking_contract(**event_kwargs)
                before_target_stat = target_path.stat()
                staged_target = target_path.with_name(".target_next.json")
                staged_target.write_text('{"revision": "b"}', encoding="utf-8")
                os.utime(staged_target, ns=(
                    before_target_stat.st_atime_ns, before_target_stat.st_mtime_ns,
                ))
                os.replace(staged_target, target_path)
                score_owner.load_selection_point_in_time_ranking_contract(**event_kwargs)
                after_target_stat = target_path.stat()
                check_true("pit_cached_ranking_rechecks_republished_target_without_pit_change",
                           canonical_gate.call_count == 2
                           and before_target_stat.st_size == after_target_stat.st_size
                           and before_target_stat.st_mtime_ns == after_target_stat.st_mtime_ns)
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
            dataset_summary_path.write_text('{"revision": "a"}', encoding="utf-8")
            def _upstream_changes_during_ranking_validation(*_args, **_kwargs):
                dataset_summary_path.write_text('{"revision": "b"}', encoding="utf-8")
                return complete
            with patch.object(
                score_owner, "_load_selection_point_in_time_ranking_contract_uncached",
                side_effect=_upstream_changes_during_ranking_validation,
            ):
                try:
                    score_owner.load_selection_point_in_time_ranking_contract(**kwargs)
                except RuntimeError as exc:
                    upstream_race_rejected = "驗證期間更新" in str(exc)
                else:
                    upstream_race_rejected = False
            check_true("pit_upstream_publication_during_contract_validation_fails_closed",
                       upstream_race_rejected)
            dataset_summary_path.unlink()
            score_owner.clear_selection_point_in_time_ranking_contract_cache()
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
            # Revisions can be published by another process, so this reader
            # cannot rely exclusively on the local producer's cache-clear call.
            import os
            previous_audit_stat = audit_file.stat()
            audit["metrics"]["pass_only_target"]["global_spearman"] = 0.0
            staged_audit = directory / ".synthetic_pit_audit_next.json"
            staged_audit.write_text(json.dumps(audit), encoding="utf-8")
            os.utime(staged_audit, ns=(
                previous_audit_stat.st_atime_ns, previous_audit_stat.st_mtime_ns,
            ))
            os.replace(staged_audit, audit_file)
            current_audit_stat = audit_file.stat()
            same_mtime_and_size = (
                previous_audit_stat.st_mtime_ns == current_audit_stat.st_mtime_ns
                and previous_audit_stat.st_size == current_audit_stat.st_size
            )
            try:
                score_owner.load_selection_point_in_time_ranking_contract(
                    **kwargs, require_model_validation_pass=True,
                )
            except ValueError as exc:
                stale_gate_rejected = "模型驗證未通過" in str(exc)
            else:
                stale_gate_rejected = False
            check_true("pit_external_audit_publication_invalidates_cached_pass_gate",
                       stale_gate_rejected and same_mtime_and_size)
            audit["metrics"]["pass_only_target"]["global_spearman"] = 0.2
            audit_file.write_text(json.dumps(audit), encoding="utf-8")
            check_true("pit_audit_republication_restores_validated_gate_without_clear",
                       score_owner.load_selection_point_in_time_ranking_contract(
                           **kwargs,
                       ).model_validation_gate["status"] == "PASS")

            # Canonical PIT access must not preserve a second fixed-key Score
            # table after a valid same-path Score + Manifest + Audit publication.
            # Use the production resolvers with isolated fixture paths.
            with patch.object(score_owner, "resolve_selection_point_in_time_score_path", return_value=score), patch.object(
                score_owner, "resolve_selection_point_in_time_manifest_path", return_value=manifest_file,
            ), patch.object(
                score_owner, "resolve_selection_point_in_time_coverage_path", return_value=coverage,
            ), patch.object(
                score_owner, "resolve_selection_point_in_time_audit_json_path", return_value=audit_file,
            ):
                canonical_kwargs = {key: value for key, value in kwargs.items()
                                    if key != "point_in_time_dir_override"}
                previous_table = score_owner.load_selection_point_in_time_score_table(
                    **canonical_kwargs,
                )
                score.write_text(
                    score.read_text(encoding="utf-8").replace(",0.7,", ",0.9,"),
                    encoding="utf-8",
                )
                manifest["artifacts"]["scores"] = _record(score)
                manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
                audit["source_artifacts"]["point_in_time_scores"] = _record(score)
                audit["source_artifacts"]["point_in_time_manifest"] = _record(manifest_file)
                audit_file.write_text(json.dumps(audit), encoding="utf-8")
                republished_table = score_owner.load_selection_point_in_time_score_table(
                    **canonical_kwargs,
                )
                check_true("pit_canonical_table_reads_valid_republished_score_without_clear",
                           float(previous_table.loc[("0050", "2021-01-04"), "breakout_quality_score"]) == 0.7
                           and float(republished_table.loc[("0050", "2021-01-04"), "breakout_quality_score"]) == 0.9)
                # A score-only change without its manifest hash must fail,
                # not return an earlier cached table or an unverified 0.9.
                score.write_text("tampered", encoding="utf-8")
                try:
                    score_owner.load_selection_point_in_time_score_table(**canonical_kwargs)
                except ValueError as exc:
                    invalid_score_rejected = "SHA256" in str(exc)
                else:
                    invalid_score_rejected = False
                check_true("pit_canonical_table_rejects_score_manifest_sha_drift_without_clear",
                           invalid_score_rejected)

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
        check_true("robustness_stage_owner_distinguishes_audit_only_and_score_build",
            not forward.score_build_required
            and not rolling.score_build_required
            and not forward.evidence_refresh_required
            and audit_only.evidence_refresh_required
            and not audit_only.score_build_required)
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
        check_true("robustness_invalid_score_requires_canonical_score_build",
            stale.score_build_required and stale.evidence_refresh_required)
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
        # Regression: Forward per-seed report absence must not erase a
        # candidate Rolling first-fold FIT from the stage projection.
        pending_rolling_fit = ModelRobustnessSeedSource(
            202, None, Path("reports/202.json"), "Forward SCORE/report missing",
            rolling_fit_candidate=True,
        )
        forward_verify = resolve_model_robustness_readiness(mode="forward", sources=(
            forward_seeds[0], pending_rolling_fit,
        ))
        pending_stage = forward_verify.seeds[1].readiness
        check_true("forward_robustness_missing_report_with_rolling_fit_shows_verify_not_build",
            forward_verify.display_action == "VERIFY FIT"
            and "1/2" in forward_verify.display_reason
            and "FIT candidates" in forward_verify.display_reason
            and pending_stage.display_action == "VERIFY FIT"
            and tuple(pending_stage.plan.action_by_key[key].action
                      for key in ("FIT", "SCORE", "EVIDENCE"))
                == ("RESUME", "BUILD", "DERIVE")
            and not forward_verify.evidence_ready)
        all_pending = resolve_model_robustness_readiness(mode="forward", sources=(
            ModelRobustnessSeedSource(101, None, Path("reports/101.json"), "missing",
                                      rolling_fit_candidate=True),
            pending_rolling_fit,
        ))
        check_true("forward_robustness_all_rolling_first_folds_present_requires_proof",
            all_pending.display_action == "VERIFY FIT"
            and "2/2" in all_pending.display_reason
            and all(item.readiness.plan.action_by_key["FIT"].action == "RESUME"
                    for item in all_pending.seeds))
        mixed = resolve_model_robustness_readiness(mode="forward", sources=(
            ModelRobustnessSeedSource(101, None, Path("reports/101.json"), "no candidate"),
            pending_rolling_fit,
        ))
        check_true("forward_robustness_partial_fit_candidates_do_not_hide_missing_fit",
            mixed.display_action == "BUILD/REFRESH"
            and "1/2" in mixed.display_reason
            and mixed.seeds[0].readiness.plan.action_by_key["FIT"].action == "BUILD")
        try:
            resolve_model_robustness_readiness(mode="rolling", sources=(
                forward_seeds[0], forward_seeds[0],
            ))
        except ValueError as exc:
            duplicate_rejected = "unique" in str(exc)
        else:
            duplicate_rejected = False
        check_true("robustness_duplicate_seed_never_merges_provenance", duplicate_rejected)
    # Seed-specific FIT candidate paths must use the same canonical FIT
    # locator as ordinary Forward, not a second robustness presence policy.
    from services.breakout_quality.fitted_model_artifacts import (
        FittedCandidateProbe, probe_fitted_candidates,
    )
    with TemporaryDirectory() as td:
        root_dir = Path(td)
        forward_model_dir = root_dir / "isolated_forward" / "seed_101"
        forward_report_path = root_dir / "isolated_forward_reports" / "seed_101" / "report.json"
        rolling_folds = root_dir / "isolated_rolling" / "seed_101" / "folds"
        first_fold = rolling_folds / "fold_20210101_20211231"
        first_fold.mkdir(parents=True)
        (first_fold / "model.pt").write_bytes(b"synthetic-provisional-weight")
        with patch("services.breakout_quality.fitted_model_artifacts.get_breakout_quality_rolling_test_mode",
                   return_value=SimpleNamespace(score_start_date="2021-01-01")):
            candidate = probe_fitted_candidates(
                project_root=root_dir, filter_id="breakout_quality_v1",
                model_architecture="inception_time_v1",
                experiment_profile="daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
                report_filename="report.json",
                forward_model_dir_override=forward_model_dir,
                forward_report_path_override=forward_report_path,
                rolling_folds_root_override=rolling_folds,
            )
            absent = probe_fitted_candidates(
                project_root=root_dir, filter_id="breakout_quality_v1",
                model_architecture="inception_time_v1",
                experiment_profile="daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
                report_filename="report.json",
                rolling_folds_root_override=root_dir / "isolated_rolling" / "seed_202" / "folds",
            )
        check_true("canonical_fit_probe_uses_exact_robustness_seed_isolation",
            candidate.rolling_first_fold_candidate
            and not candidate.forward_fit_candidate
            and not absent.rolling_first_fold_candidate)
        from services.research import breakout_quality_application as candidate_app
        candidate_settings = SimpleNamespace(
            filter_id="breakout_quality_v1", model_architecture="inception_time_v1",
            experiment_profile="daily_universal_full_horizon_no_breach_full_list_ndcg_pairwise",
        )
        def candidate_dirs(_settings, *, seed, rolling):
            if rolling:
                return root_dir / "isolated_rolling" / f"seed_{seed}", root_dir / "unused"
            return root_dir / "isolated_forward" / f"seed_{seed}", root_dir / "isolated_forward_reports" / f"seed_{seed}"
        with (
            patch.object(candidate_app, "_model_robustness_dirs", side_effect=candidate_dirs),
            patch.object(candidate_app, "_load_forward_robustness_seed",
                         return_value=(None, forward_report_path, "Forward report missing")),
            patch.object(candidate_app, "_probe_continuous_fitted_candidates",
                         return_value=FittedCandidateProbe(False, True)) as status_probe,
        ):
            observed = candidate_app._observe_model_robustness_seed(
                candidate_settings, seed=101, rolling=False,
            )
        check_true("forward_robustness_application_probes_exact_seed_fold_when_report_missing",
            observed.seed == 101 and observed.rolling_fit_candidate
            and observed.verified_evidence is None
            and status_probe.call_count == 1
            and status_probe.call_args.kwargs["rolling_folds_root_override"] == rolling_folds
            and status_probe.call_args.kwargs["forward_model_dir_override"] == forward_model_dir)
    # A source snapshot and its dependency plan must agree before any Producer
    # can run. Source order alone is not sufficient if a stage owner is bypassed
    # or accidentally returns another benchmark's seed plan.
    from services.research import breakout_quality_application as snapshot_app
    with (
        patch.object(snapshot_app, "_observe_model_robustness_seed", side_effect=lambda _settings, *, seed, rolling:
                     ModelRobustnessSeedSource(int(seed), good, Path(f"synthetic/{seed}.json"), None)),
        patch.object(snapshot_app, "resolve_model_robustness_readiness", return_value=SimpleNamespace(
            seeds=(SimpleNamespace(seed=202), SimpleNamespace(seed=101)),
        )),
    ):
        try:
            snapshot_app._plan_model_robustness_snapshot(
                SimpleNamespace(), seeds=(101, 202), rolling=False,
            )
        except RuntimeError as exc:
            mismatch_rejected = "plan seed order/identity" in str(exc)
        else:
            mismatch_rejected = False
    check_true("source_plan_snapshot_rejects_cross_seed_plan_pairing", mismatch_rejected)

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
                                     after_first_build_stale, final_stale_first=False,
                                     seeds=(101, 202), upstream_executes=True,
                                     rolling=False):
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
        def prepare(_name, _settings, *, execution_observer=None):
            phase["prepared"] = bool(upstream_executes)
            if execution_observer is not None:
                execution_observer(bool(upstream_executes))
            return 0
        def build(_name, _settings, *, seed, **_kwargs):
            phase["built"].append(int(seed))
            return 0
        with (
            patch("services.breakout_quality.model_artifact_readiness.standard_model_sop_completeness_issues", return_value=()),
            patch.object(model_app, "get_breakout_quality_model_test_settings",
                         return_value=SimpleNamespace(model_profiles=(("MR-SYN", "synthetic_profile"),))),
            patch.object(model_app, "get_breakout_quality_workflow_settings", return_value=workflow),
            patch.object(model_app, "resolve_robustness_benchmark_seeds", return_value=tuple(seeds)),
            patch.object(model_app, "_observe_model_robustness_seed", side_effect=observe),
            patch.object(model_app, "_prepare_continuous_research_inputs", side_effect=prepare) as prep,
            patch.object(model_app, "_build_forward_robustness_seed", side_effect=build) as fwd_producer,
            patch.object(model_app, "_build_rolling_robustness_seed", side_effect=build) as roll_producer,
            patch.object(model_app, "_load_forward_robustness_seed",
                         side_effect=lambda _settings, *, seed: (synthetic_report, Path(f"synthetic/{seed}.json"), None)),
            patch.object(model_app, "_load_rolling_robustness_seed",
                         side_effect=lambda _settings, *, seed: (synthetic_report, Path(f"synthetic/{seed}.json"), None)),
            patch.object(model_app, "invalidate_rolling_model_source_cache"),
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
                status = model_app._run_configured_model_robustness("synthetic", rolling=rolling)
                error = None
            except RuntimeError as exc:
                status, error = None, str(exc)
        producer = roll_producer if rolling else fwd_producer
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

    # The preparation API must report the canonical orchestrator's actual
    # executed actions, not infer mutations from a successful return code.
    for executed in ((), (("dataset:synthetic", "BUILD", None, "producer"),)):
        observations = []
        with (
            patch.object(model_app, "run_research_artifact_preparation", return_value=SimpleNamespace(
                plan=SimpleNamespace(actions=()), executed=executed,
            )),
            patch.object(model_app, "get_breakout_quality_experiment_profile", return_value=SimpleNamespace(
                training_sample_scope="synthetic_not_daily",
            )),
            redirect_stdout(StringIO()),
        ):
            prepared_status = model_app._prepare_continuous_research_inputs(
                "synthetic", workflow, execution_observer=observations.append,
            )
        check_true(f"phase12_preparation_observer_matches_executed_{bool(executed)}",
                   prepared_status == 0 and observations == [bool(executed)])

    observer_on_error = []
    with (
        patch.object(model_app, "run_research_artifact_preparation", side_effect=RuntimeError("blocked")),
        redirect_stdout(StringIO()),
    ):
        error_status = model_app._prepare_continuous_research_inputs(
            "synthetic", workflow, execution_observer=observer_on_error.append,
        )
    check_true("phase12_failed_preparation_does_not_signal_successful_noop",
               error_status == 2 and observer_on_error == [])

    # Phase 12: the canonical preparation outcome distinguishes a real
    # Dataset/Target publication from a read-only upstream REUSE check. The
    # latter must not pay for a second complete seed source-plan snapshot.
    four_seeds = (101, 202, 303, 404)
    one_missing = {101: False, 202: True, 303: True, 404: True}
    for test_rolling in (False, True):
        status, no_op, prep_count, build_count, report_count, agg_count, error = run_phase9_synthetic_routing(
            initially_ready=one_missing, after_prepare_ready=one_missing,
            after_first_build_stale=False, seeds=four_seeds,
            upstream_executes=False, rolling=test_rolling,
        )
        changed_status, changed, changed_prep, changed_build, changed_report, changed_agg, changed_error = run_phase9_synthetic_routing(
            initially_ready=one_missing, after_prepare_ready=one_missing,
            after_first_build_stale=False, seeds=four_seeds,
            upstream_executes=True, rolling=test_rolling,
        )
        name = "rolling" if test_rolling else "forward"
        check_true(f"phase12_{name}_noop_upstream_saves_one_full_seed_observation",
            status == changed_status == 0 and error is changed_error is None
            and prep_count == changed_prep == build_count == changed_build == 1
            and report_count == changed_report == agg_count == changed_agg == 1
            and no_op["built"] == changed["built"] == [101]
            and len(no_op["observed"]) == 11
            and len(changed["observed"]) == 15
            and all(row[1] is False for row in no_op["observed"])
            and any(row[1] is True for row in changed["observed"])
        )

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

    # Single Forward/Rolling model comparisons share the same cross-model
    # publication guarantee as Robustness. Only synthetic sources and producer
    # ports are used; no market Data, PIT score or artifact is touched.
    def run_single_comparison_revalidation(
        *, rolling: bool, stale_previous: bool = False,
        all_ready: bool = False, blocked_second: bool = False,
        reused_first: bool = False,
    ):
        profiles = (("MR-SYN-A", "phase11_a"), ("MR-SYN-B", "phase11_b"))
        state = {
            "ready": {"phase11_a": all_ready or reused_first, "phase11_b": all_ready},
            "version": {"phase11_a": 0, "phase11_b": 0},
            "built": [], "observed": [],
        }
        def workflow(*, experiment_profile):
            return SimpleNamespace(experiment_profile=str(experiment_profile))
        def evidence(profile):
            return {"standard_model_sop": {"model": profile, "version": state["version"][profile]}}
        def contract(profile):
            return SimpleNamespace(report=evidence(profile), audit_path=Path(f"synthetic/{profile}/audit.json"))
        def observe(settings, *, rolling, mode=None):
            profile = settings.experiment_profile
            ready = bool(state["ready"][profile])
            state["observed"].append((profile, ready, tuple(state["built"])))
            return SimpleNamespace(
                evidence_ready=ready, source_contract=contract(profile) if ready else None,
                evidence_payload=evidence(profile) if ready else None,
                display_action="REUSE" if ready else "BUILD",
                display_reason="READY" if ready else "stale",
                source_reason=None if ready else "stale",
            )
        def produce(_program, settings, **_kwargs):
            profile = settings.experiment_profile
            state["built"].append(profile)
            state["ready"][profile] = True
            state["version"][profile] += 1
            if profile == "phase11_a":
                state["ready"]["phase11_b"] = False
            else:
                state["ready"]["phase11_a"] = not stale_previous
                state["version"]["phase11_a"] += 1
            return 0 if rolling else (0, contract(profile), "BUILD/REFRESH")
        def inputs(settings):
            blocked = bool(blocked_second and settings.experiment_profile == "phase11_b")
            return SimpleNamespace(
                blocked=blocked,
                actions=(SimpleNamespace(action="BLOCKED", description="synthetic blocker"),) if blocked else (),
            )
        rendered = []
        def render(rows, **_kwargs):
            rendered.extend((row["model_id"], dict(row["payload"]["standard_model_sop"])) for row in rows)
            return "SYNTHETIC"
        with (
            patch.object(model_app, "get_breakout_quality_model_test_settings",
                         return_value=SimpleNamespace(model_profiles=profiles)),
            patch.object(model_app, "get_breakout_quality_workflow_settings", side_effect=workflow),
            patch.object(model_app, "_recover_pending_model_fit_repairs", return_value=0),
            patch.object(model_app, "get_breakout_quality_rolling_test_mode", return_value=SimpleNamespace()),
            patch.object(model_app, "_resolve_continuous_model_readiness", side_effect=observe),
            patch.object(model_app, "_collect_continuous_research_input_plan", side_effect=inputs),
            patch.object(model_app, "_load_canonical_forward_checkpoint_import_source", return_value=(None, None)),
            patch.object(model_app, "_run_continuous_rolling_mode_direct", side_effect=produce) as roll_producer,
            patch.object(model_app, "_load_reusable_rolling_standard_report", side_effect=lambda settings, _mode: (
                contract(settings.experiment_profile), evidence(settings.experiment_profile), None,
            )),
            patch.object(model_app, "_ensure_continuous_forward_model_report", side_effect=produce) as fwd_producer,
            patch.object(model_app, "_render_standard_model_comparison", side_effect=render),
            patch.object(model_app, "_write_standard_model_comparison_report",
                         return_value=(Path("synthetic.json"), Path("synthetic.md"))) as publish,
            patch.object(model_app, "render_status_paths", return_value=""),
            patch.object(model_app, "_print_model_action_status"),
            patch.object(model_app, "invalidate_rolling_model_source_cache") as clear_cache,
            redirect_stdout(StringIO()),
        ):
            try:
                code = model_app._run_configured_model_comparison("synthetic", rolling=rolling)
                error = None
            except RuntimeError as exc:
                code, error = None, str(exc)
        return (code, error, state, rendered, publish.call_count,
                clear_cache.call_count, roll_producer.call_count, fwd_producer.call_count)

    for rolling in (False, True):
        label = "rolling" if rolling else "forward"
        code, error, state, rows, publishes, clears, roll_calls, fwd_calls = (
            run_single_comparison_revalidation(rolling=rolling)
        )
        check_true(f"single_{label}_replans_next_model_and_refreshes_earlier_evidence",
                   code == 0 and error is None and publishes == 1
                   and state["built"] == ["phase11_a", "phase11_b"]
                   and ("phase11_b", False, ("phase11_a",)) in state["observed"]
                   and rows[0][1]["version"] == 2
                   and (roll_calls, fwd_calls) == ((2, 0) if rolling else (0, 2))
                   and clears == (3 if rolling else 0))
        code, error, state, rows, publishes, *_ = run_single_comparison_revalidation(
            rolling=rolling, stale_previous=True,
        )
        check_true(f"single_{label}_later_producer_cannot_publish_stale_earlier_source",
                   code is None and publishes == 0 and not rows
                   and state["built"] == ["phase11_a", "phase11_b"]
                   and "MR-SYN-A" in str(error) and "拒絕發布報表" in str(error))
        code, error, state, rows, publishes, clears, roll_calls, fwd_calls = (
            run_single_comparison_revalidation(rolling=rolling, all_ready=True)
        )
        check_true(f"single_{label}_all_ready_keeps_one_observation_per_model",
                   code == 0 and error is None and publishes == 1
                   and len(state["observed"]) == 2 and not state["built"]
                   and (roll_calls, fwd_calls) == (0, 0)
                   and clears == (1 if rolling else 0))
        code, error, state, rows, publishes, _, roll_calls, fwd_calls = (
            run_single_comparison_revalidation(rolling=rolling, blocked_second=True)
        )
        check_true(f"single_{label}_atomic_blocker_preflight_runs_before_any_producer",
                   code == 2 and error is None and publishes == 0
                   and not state["built"] and (roll_calls, fwd_calls) == (0, 0))
        code, error, state, rows, publishes, *_ = run_single_comparison_revalidation(
            rolling=rolling, reused_first=True, stale_previous=True,
        )
        check_true(f"single_{label}_previous_reused_model_also_requires_publication_revalidation",
                   code is None and publishes == 0 and not rows
                   and state["built"] == ["phase11_b"]
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
    # Dataset content integrity and PIT upstream version invalidation share the
    # canonical readiness owner. These are deliberately tiny fake artifact bytes:
    # no new Dataset/Target scientific identity or training fixture is produced.
    from core.breakout_quality_policy import get_breakout_quality_model_research_settings
    from filters.breakout_quality import dataset_readiness as readiness_owner
    from filters.breakout_quality import dataset_store as dataset_owner
    from filters.breakout_quality import artifact_dependency_registry as upstream_owner
    from filters.breakout_quality.paths import resolve_filter_output_dir
    from filters.breakout_quality.artifacts import build_file_manifest
    from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY, FEATURE_COLUMNS, CONTEXT_COLUMNS
    import os
    integrity_settings = get_breakout_quality_model_research_settings()
    with TemporaryDirectory() as td:
        project = Path(td)
        dataset_paths = dataset_owner.resolve_dataset_paths(
            resolve_filter_output_dir(project, integrity_settings.filter_id)
        )
        dataset_paths.output_dir.mkdir(parents=True)
        for name, artifact in dataset_paths.artifact_paths().items():
            artifact.write_bytes(f"{name} old bytes".encode())
        inventory = {"csv_inventory_sha256": "synthetic-source-bytes"}
        summary_data = {
            "filter_id": integrity_settings.filter_id,
            "dataset": "full",
            "dataset_storage_schema_version": dataset_owner.DATASET_STORAGE_SCHEMA_VERSION,
            "dataset_storage_format": dataset_owner.DATASET_STORAGE_FORMAT,
            "source_selection": {"requested_max_tickers": 0},
            "feature_cache_policy": DEFAULT_LABEL_POLICY.feature_cache_manifest_payload(),
            "feature_columns": list(FEATURE_COLUMNS),
            "context_columns": list(CONTEXT_COLUMNS),
            "label_policy": DEFAULT_LABEL_POLICY.label_manifest_payload(),
            "policy": DEFAULT_LABEL_POLICY.as_manifest_payload(),
            "source_data_inventory": inventory,
            "dataset_artifacts": {
                name: build_file_manifest(path)
                for name, path in dataset_paths.artifact_paths().items()
            },
        }
        dataset_paths.summary.write_text(json.dumps(summary_data), encoding="utf-8")
        readiness_args = {
            "filter_id": integrity_settings.filter_id,
            "model_architecture": integrity_settings.model_architecture,
            "dataset": "full", "max_tickers": 0,
        }
        signature_args = {
            "filter_id": integrity_settings.filter_id,
            "model_architecture": integrity_settings.model_architecture,
            "experiment_profile": integrity_settings.experiment_profile,
        }
        with patch.object(readiness_owner, "build_source_data_inventory", return_value=inventory):
            before = readiness_owner.collect_dataset_readiness(project, **readiness_args)
            check_true("dataset_existing_sha_records_ready", before.ready)
            signature_before = upstream_owner.model_upstream_publication_signature(
                project, **signature_args,
            )
            artifact = dataset_paths.feature_bank
            old_stat = artifact.stat()
            staged = artifact.with_name(".synthetic_replacement.npy")
            staged.write_bytes(b"X" + artifact.read_bytes()[1:])
            os.utime(staged, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
            os.replace(staged, artifact)
            signature_after = upstream_owner.model_upstream_publication_signature(
                project, **signature_args,
            )
            check_true("dataset_rewrite_without_summary_invalidates_pit_hint",
                       signature_before != signature_after
                       and old_stat.st_size == artifact.stat().st_size
                       and old_stat.st_mtime_ns == artifact.stat().st_mtime_ns)
            stale = readiness_owner.collect_dataset_readiness(project, **readiness_args)
            check_true("dataset_same_metadata_corruption_rejected_by_canonical_readiness",
                       not stale.ready and any("feature_bank artifact SHA256 mismatch" in reason
                                                   for reason in stale.reasons))
            summary_data["dataset_artifacts"]["feature_bank"] = build_file_manifest(artifact)
            dataset_paths.summary.write_text(json.dumps(summary_data), encoding="utf-8")
            fresh = readiness_owner.collect_dataset_readiness(project, **readiness_args)
            check_true("dataset_legit_summary_republication_allows_ready", fresh.ready)
            with patch.object(dataset_owner, "compute_file_sha256", side_effect=AssertionError(
                "unchanged physical revision should hit bounded SHA cache"
            )):
                check_true("dataset_unchanged_revision_does_not_rehash",
                           readiness_owner.collect_dataset_readiness(project, **readiness_args).ready)
            # Market-set input is architecture-dependent, but its evidence
            # must use the same Dataset content-verification owner.
            from filters.breakout_quality.models.architectures import SUPPORTED_MODEL_ARCHITECTURES
            from filters.breakout_quality.models.spec import get_model_spec
            from filters.breakout_quality.market_set import market_set_contract_payload
            market_arch = next(
                arch for arch in SUPPORTED_MODEL_ARCHITECTURES
                if get_model_spec(arch).requires_market_set
            )
            for name, market_path in dataset_paths.market_set_artifact_paths().items():
                market_path.write_bytes(f"{name} valid market".encode())
            summary_data["market_set_contract"] = market_set_contract_payload()
            summary_data["market_set_artifacts"] = {
                name: build_file_manifest(path)
                for name, path in dataset_paths.market_set_artifact_paths().items()
            }
            dataset_paths.summary.write_text(json.dumps(summary_data), encoding="utf-8")
            market_args = {**readiness_args, "model_architecture": market_arch}
            check_true("market_set_artifacts_sha_ready_when_valid",
                       readiness_owner.collect_dataset_readiness(project, **market_args).ready)
            market_feature = dataset_paths.market_daily_features
            market_feature.write_bytes(b"X" + market_feature.read_bytes()[1:])
            invalid_market = readiness_owner.collect_dataset_readiness(project, **market_args)
            check_true("market_set_artifacts_sha_mismatch_rejects_ready",
                       not invalid_market.ready and any(
                           "market_daily_features artifact SHA256 mismatch" in reason
                           for reason in invalid_market.reasons
                       ))
            # An interrupted, same-size rewrite *during* the SHA check cannot
            # be persisted as a reusable verified revision.
            artifact.write_bytes(b"Z" + artifact.read_bytes()[1:])
            real_hash = dataset_owner.compute_file_sha256
            def mutation_during_hash(path):
                observed = real_hash(path)
                artifact.write_bytes(b"Y" + artifact.read_bytes()[1:])
                return observed
            with patch.object(dataset_owner, "compute_file_sha256", side_effect=mutation_during_hash):
                raced = readiness_owner.collect_dataset_readiness(project, **readiness_args)
            check_true("dataset_changed_during_sha_validation_rejected",
                       not raced.ready and any("verification failed" in reason
                                               for reason in raced.reasons))

    # The Continuous Target manifest loader already owns main-target hashes.
    # Optional published components must not silently bypass that same owner.
    from filters.breakout_quality.continuous_target import (
        CONTINUOUS_TARGET_SCHEMA_VERSION, TARGET_FAVORABLE_RETURN_FILENAME,
        TARGET_RAW_FILENAME, TARGET_VALID_MASK_FILENAME,
        load_validated_continuous_target_manifest, resolve_continuous_target_dir,
    )
    from core.breakout_quality_runtime import STRATEGY_ALIGNED_TARGET_ID
    with TemporaryDirectory() as td:
        project = Path(td)
        target_dir = resolve_continuous_target_dir(
            project, integrity_settings.filter_id, target_id=STRATEGY_ALIGNED_TARGET_ID,
        )
        target_dir.mkdir(parents=True)
        target_files = {
            "target_raw_r": target_dir / TARGET_RAW_FILENAME,
            "valid_mask": target_dir / TARGET_VALID_MASK_FILENAME,
            "favorable_return": target_dir / TARGET_FAVORABLE_RETURN_FILENAME,
        }
        for name, path in target_files.items():
            path.write_bytes(f"{name} valid target".encode())
        target_manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": integrity_settings.filter_id,
            "target_contract": {"target_id": STRATEGY_ALIGNED_TARGET_ID},
            "artifacts": {name: build_file_manifest(path) for name, path in target_files.items()},
        }
        (target_dir / "manifest.json").write_text(json.dumps(target_manifest), encoding="utf-8")
        load_args = {"target_id": STRATEGY_ALIGNED_TARGET_ID}
        check_true("target_published_component_valid_before_rewrite",
                   isinstance(load_validated_continuous_target_manifest(
                       project, integrity_settings.filter_id, **load_args,
                   ), dict))
        target_files["favorable_return"].write_bytes(
            b"X" + target_files["favorable_return"].read_bytes()[1:]
        )
        try:
            load_validated_continuous_target_manifest(
                project, integrity_settings.filter_id, **load_args,
            )
            accepted_changed_component = True
        except ValueError as exc:
            accepted_changed_component = "component SHA256" not in str(exc)
        check_true("target_published_component_corruption_rejected",
                   not accepted_changed_component)

    # A derived Target pins another Target's manifest by SHA. PIT cache keys
    # must notice that *linked* publication, not just the derived arrays.
    # This uses the real Target provenance validator and existing PIT
    # upstream-version owner; no independent source-legality rule is added.
    from core.breakout_quality_registry import STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE
    from core.breakout_quality_runtime import STRATEGY_ALIGNED_NO_TIME_TARGET_ID
    with TemporaryDirectory() as td:
        project = Path(td)
        filter_id = integrity_settings.filter_id
        original_dir = resolve_continuous_target_dir(
            project, filter_id, target_id=STRATEGY_ALIGNED_TARGET_ID,
        )
        derived_dir = resolve_continuous_target_dir(
            project, filter_id, target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
        )
        original_dir.mkdir(parents=True)
        derived_dir.mkdir(parents=True)
        original_manifest = original_dir / "manifest.json"
        source_dataset_record = {
            "feature_bank": {"filename": "feature.npy", "size_bytes": 1, "sha256": "0" * 64}
        }
        source_payload = {"dataset_artifact_source": source_dataset_record, "revision": "a"}
        original_manifest.write_text(json.dumps(source_payload), encoding="utf-8")
        derived_artifacts = {}
        for name, filename in (
            ("target_raw_r", TARGET_RAW_FILENAME),
            ("valid_mask", TARGET_VALID_MASK_FILENAME),
        ):
            path = derived_dir / filename
            path.write_bytes(f"isolated-{name}".encode())
            derived_artifacts[name] = build_file_manifest(path)
        derived_manifest = {
            "schema_version": CONTINUOUS_TARGET_SCHEMA_VERSION,
            "filter_id": filter_id,
            "target_contract": {"target_id": STRATEGY_ALIGNED_NO_TIME_TARGET_ID},
            "source_target": {
                "target_id": STRATEGY_ALIGNED_TARGET_ID,
                "manifest": build_file_manifest(original_manifest),
            },
            "artifacts": derived_artifacts,
        }
        (derived_dir / "manifest.json").write_text(
            json.dumps(derived_manifest), encoding="utf-8"
        )
        source_args = {
            "filter_id": filter_id,
            "model_architecture": integrity_settings.model_architecture,
            "experiment_profile": STRATEGY_ALIGNED_NO_TIME_ALL_EVENT_MSE_PROFILE,
        }
        def published_signature():
            return upstream_owner.model_upstream_publication_signature(project, **source_args)
        def original_source_valid():
            return load_validated_continuous_target_manifest(
                project, filter_id, target_id=STRATEGY_ALIGNED_NO_TIME_TARGET_ID,
                expected_dataset_artifacts=source_dataset_record,
            )
        check_true("derived_target_initial_linked_source_valid",
                   bool(original_source_valid()))
        initial_signature = published_signature()
        check("derived_target_unchanged_linked_source_stable",
              initial_signature, published_signature())
        previous_stat = original_manifest.stat()
        sibling = original_dir / "updated-manifest.tmp"
        sibling.write_text(
            json.dumps({**source_payload, "revision": "b"}), encoding="utf-8",
        )
        os.utime(sibling, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
        sibling.replace(original_manifest)
        check_true("derived_target_linked_manifest_invalidates_pit_cache",
                   initial_signature != published_signature())
        try:
            original_source_valid()
            stale_source_accepted = True
        except ValueError as exc:
            stale_source_accepted = "source manifest已改變" not in str(exc)
        check_true("derived_target_linked_source_stays_canonical_fail_closed",
                   not stale_source_accepted)
        before_missing = published_signature()
        original_manifest.unlink()
        check_true("derived_target_missing_linked_manifest_invalidates_pit_cache",
                   before_missing != published_signature())

    return results, summary
