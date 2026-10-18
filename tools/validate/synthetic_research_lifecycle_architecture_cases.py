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

    # Mutation tests deliberately introduce new business logic under *different*
    # names to catch drift not detected by a list of old legacy function labels.
    mut = dict(sources)
    mut[PIT_PRODUCER] += "\ndef fold_training_identity(contract):\n    return 'shadow'\n"
    check_true("second_pit_fit_identity_blocked", bool(check_research_fit_architecture_sources(mut)))
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
        )
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "local missing")), patch.object(
            pit, "load_forward_compatible_rolling_checkpoint", return_value=(None, "rolling missing"),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("missing_fit_action", "FIT_REQUIRED", result.source)
        check_true("missing_fit_has_evidence_reason", "rolling missing" in str(result.reason))
        (model_dir / "model.pt").write_bytes(b"unverified bytes")
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "stale")), patch.object(
            pit, "load_forward_compatible_rolling_checkpoint", return_value=(None, "not equivalent"),
        ):
            try:
                owner.resolve_forward_fitted_checkpoint(**kwargs)
                blocked = False
            except RuntimeError as exc:
                blocked = "not equivalent" in str(exc)
        check_true("unverified_existing_fit_blocks_refit", blocked)
        candidate = SimpleNamespace(selected_epoch=3, source="verified_fit")
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(candidate, None)), patch.object(
            pit, "load_forward_compatible_rolling_checkpoint", side_effect=AssertionError("must not scan Rolling when Forward FIT valid"),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("valid_forward_fit_reuse", "FORWARD_FIT", result.source)
        with patch.object(owner, "load_reusable_fitted_model_contract", return_value=(None, "old")), patch.object(
            pit, "load_forward_compatible_rolling_checkpoint", return_value=(candidate, "verified"),
        ), patch.object(owner, "load_model_from_fitted_checkpoint", return_value="model"), patch.object(
            owner, "import_verified_rolling_forward_checkpoint", return_value=(None, True),
        ):
            result = owner.resolve_forward_fitted_checkpoint(**kwargs)
        check("cross_mode_fit_reuse", "ROLLING_FIT", result.source)
        check("cross_mode_fit_loaded_without_refit", "model", result.preloaded_model)

    # No approved science/report identity may change merely because this is a
    # code-organization refactor; those fingerprints live in the report registry.
    return results, summary
