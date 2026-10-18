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
