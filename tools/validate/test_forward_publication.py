"""Isolated Forward admission/publication failures; never train or use live data."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from core.breakout_quality_policy import get_breakout_quality_model_research_settings
from core.breakout_quality_registry import get_breakout_quality_experiment_profile
from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
from filters.breakout_quality import ranking_score_store as store
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.ranker_training_contract import training_semantics
from services.breakout_quality import forward_model_source as source_owner
from services.breakout_quality import forward_publication as publication_owner
from services.breakout_quality.fitted_model_artifacts import (
    current_default_fitting_settings, write_fitted_model_contract,
    load_reusable_fitted_model_contract,
)


@contextmanager
def independent_upstream_evidence():
    """Isolate current upstream IO; persisted model/score validators remain real."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(store, "_validate_current_forward_score_source_lineage"))
        stack.enter_context(patch.object(source_owner, "_forward_source_provenance_issues", return_value=()))
        stack.enter_context(patch.object(source_owner, "standard_model_sop_completeness_issues", return_value=()))
        yield


def write_forward_fixture(model_dir, output_dir, *, profile_name=None, scores=(0.2, 0.4), seed=101, filter_id=None):
    settings = get_breakout_quality_model_research_settings()
    profile = get_breakout_quality_experiment_profile(profile_name or settings.experiment_profile)
    recipe = get_continuous_ranker_execution_recipe(profile.name)
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model_dir / "model.pt"
    model.write_bytes(b"synthetic frozen checkpoint")
    score = output_dir / store.DAILY_RANKER_OOS_SCORE_FILENAME
    frame = pd.DataFrame({"ticker": [f"T{i:04}" for i in range(len(scores))],
                          "date": ["2021-01-04"] * len(scores),
                          "group_index": list(range(len(scores))), "model_score": scores})
    frame.to_csv(score, index=False, compression="gzip", encoding="utf-8-sig")
    coverage = {"inference_eligible_groups": len(scores), "target_evaluable_groups": len(scores),
                "future_target_required_for_score": False}
    identity = {"filter_id": filter_id or settings.filter_id, "model_architecture": settings.model_architecture,
                "experiment_profile": profile.name}
    semantic = training_semantics(profile)
    eligibility = store.build_score_eligibility_contract(profile)
    manifest = {**identity, "experiment_settings": profile.as_manifest_payload(),
                "training_objective": profile.training_objective,
                "training_sample_scope": profile.training_sample_scope,
                "training_label_scope": profile.training_label_scope,
                "training_semantics": semantic, "continuous_target_id": profile.continuous_target_id,
                "score_eligibility_contract": eligibility, "forward_score_coverage": coverage,
                "model": build_file_manifest(model),
                "research_outputs": {"oos_scores_gzip": build_file_manifest(score)},
                "outer_oos_policy": {"oos_start_date": "2021-01-01", "configured_oos_end_date": "2021-12-31"},
                "model_information_cutoff": "2020-12-31"}
    report = {**identity, "experiment_settings": profile.as_manifest_payload(),
              "training": {"objective": profile.training_objective, "loss": profile.loss_name,
                           "sample_scope": profile.training_sample_scope,
                           "training_label_scope": profile.training_label_scope, **semantic,
                           "pairwise_reduction": recipe.pairwise_reduction,
                           "pair_weight_policy": recipe.objective_policy.pair_weight_policy,
                           "epoch_selection_metric": profile.epoch_selection_metric, "seed": seed,
                           "selected_epoch": 3},
              "score_eligibility_contract": eligibility, "forward_score_coverage": coverage,
              "status": "RESULT_AVAILABLE", "standard_model_sop": {"evaluation_mode": "forward_oos"},
              "artifacts": {"oos_scores_gzip": build_file_manifest(score)}}
    fitting_identity = {"training_settings": current_default_fitting_settings(), "seed": seed}
    write_fitted_model_contract(model_dir=model_dir, fitting_identity=fitting_identity,
                               selected_epoch=3, epoch_selection={}, final_refit_history=[],
                               trainable_parameter_count=1, total_parameter_count=1)
    (model_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (output_dir / store.CONTINUOUS_RANKER_REPORT_FILENAME).write_text(json.dumps(report), encoding="utf-8")
    return identity, manifest, report, fitting_identity, score


class ForwardAdmissionTests(unittest.TestCase):
    def test_canonical_primary_and_secondary_keep_admitted_snapshot(self):
        from core.buy_sort import BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0
        from core.breakout_quality_ranking_context import breakout_quality_ranking_source_context
        from filters.breakout_quality.paths import resolve_filter_artifact_paths, resolve_filter_model_output_dir
        from filters.breakout_quality.runtime import resolve_breakout_quality_candidate_rank
        from services.research.strategy_compare_runtime_contract import resolve_strategy_compare_runtime_contract
        settings = get_breakout_quality_model_research_settings()
        for changed_source in ("primary", "safety"):
            with self.subTest(changed_source=changed_source), TemporaryDirectory() as td, independent_upstream_evidence():
                root = Path(td)
                sources = {}
                for source_id, filter_id in (("primary", settings.filter_id), ("safety", "safety_fixture")):
                    model = resolve_filter_artifact_paths(root, filter_id, settings.model_architecture,
                                                          settings.experiment_profile).model_dir
                    output = resolve_filter_model_output_dir(root, filter_id, settings.model_architecture,
                                                             settings.experiment_profile)
                    identity, _, report, _, score = write_forward_fixture(model, output, filter_id=filter_id, seed=42)
                    sources[source_id] = (identity, model, output, report, score)
                primary, secondary = sources["primary"], sources["safety"]
                options = {"safety_dl_id": "safety", "safety_filter_id": secondary[0]["filter_id"],
                           "safety_model_architecture": settings.model_architecture,
                           "safety_experiment_profile": settings.experiment_profile,
                           "safety_score_source": store.SCORE_SOURCE_CONTINUOUS_RANKER_OOS}
                with patch.object(source_owner, "_forward_source_provenance_issues", return_value=()) as scope:
                    runtime = resolve_strategy_compare_runtime_contract(
                        root=root, dataset="reduced", comparison_mode="score-ranking",
                        ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
                        ranking_options=options, optional_entry_filter_policy="current",
                        score_source=store.SCORE_SOURCE_CONTINUOUS_RANKER_OOS, **primary[0],
                        threshold=None, hard_filter_source=None)
                self.assertEqual([c.kwargs["dataset"] for c in scope.call_args_list], ["reduced", "reduced"])
                self.assertIsNone(runtime.ranking_source["score_path_override"])
                self.assertEqual(runtime.ranking_options, options)
                self.assertEqual(len(runtime.score_bindings), 2)
                with breakout_quality_ranking_source_context(**runtime.ranking_source, score_bindings=runtime.score_bindings):
                    args = dict(project_root=str(root), filter_id=primary[0]["filter_id"], ticker="T0001",
                                signal_date="2021-01-04", information_date="2021-01-04", high_len=20)
                    original = resolve_breakout_quality_candidate_rank(**args)
                    self.assertEqual(original["score"], 0.4)
                    self.assertEqual(original["safety_score"], 0.4)
                    identity, model, output, _, _ = sources[changed_source]
                    # A complete, internally valid newer seed replaces the source.
                    write_forward_fixture(model, output, filter_id=identity["filter_id"], seed=202)
                    with self.assertRaisesRegex(ValueError, "changed"):
                        resolve_breakout_quality_candidate_rank(**args)

    def test_training_ready_uses_explicit_prepared_dataset(self):
        from services.research.strategy_compare_training import validate_strategy_compare_training_artifacts
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            identity, _, _, _, _ = write_forward_fixture(root / "model", root / "output")
            with patch.object(source_owner, "_forward_source_provenance_issues", return_value=()) as source:
                validate_strategy_compare_training_artifacts(
                    project_root=root, source=SimpleNamespace(**identity, score_source=store.SCORE_SOURCE_CONTINUOUS_RANKER_OOS),
                    workflow=SimpleNamespace(strategy_dataset="full"), dataset="reduced", seed=101,
                    model_dir=root / "model", research_dir=root / "output",
                    comparison_start="2021-01-04", comparison_end="2021-01-04")
            self.assertEqual(source.call_args.kwargs["dataset"], "reduced")

    def test_persisted_fit_identity_cannot_disagree_with_verified_report(self):
        from services.breakout_quality import fitted_model_artifacts as fit_owner
        with TemporaryDirectory() as td:
            root = Path(td)
            identity, manifest, report, _, _ = write_forward_fixture(root / "model", root / "output")
            path = root / "model" / "fitted_model_manifest.json"
            baseline = json.loads(path.read_text(encoding="utf-8"))
            with patch.object(fit_owner, "forward_published_source_provenance_issues", return_value=()):
                def issues():
                    return fit_owner.forward_fitted_source_provenance_issues(
                        model_dir=root / "model", model_manifest=manifest, report=report,
                        dataset_summary={"source_data_inventory": {}, "dataset_artifacts": {}})
                self.assertEqual(issues(), ())
                for field, value in (("seed", 202), ("training_objective", "wrong"),
                                     ("experiment_settings", {"fit_contract_version": 999})):
                    with self.subTest(field=field):
                        changed = deepcopy(baseline)
                        changed["fitting_identity"][field] = value
                        path.write_text(json.dumps(changed), encoding="utf-8")
                        self.assertTrue(any(field in text for text in issues()))
                changed = deepcopy(baseline)
                changed["fitting_result"]["selected_epoch"] = 99
                path.write_text(json.dumps(changed), encoding="utf-8")
                self.assertTrue(any("selected_epoch" in text for text in issues()))

    def test_complete_isolated_ready_mutations_and_artifact_projection(self):
        from services.research.strategy_compare_training import validate_strategy_compare_training_artifacts
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            model, output = root / "model", root / "output"
            identity, manifest, report, fit, score = write_forward_fixture(model, output)
            def validate():
                return validate_strategy_compare_training_artifacts(dataset="full",
                    project_root=root, source=SimpleNamespace(**identity, score_source=store.SCORE_SOURCE_CONTINUOUS_RANKER_OOS),
                    workflow=SimpleNamespace(), seed=101, model_dir=model, research_dir=output,
                    comparison_start="2021-01-04", comparison_end="2021-01-04")
            accepted = validate()
            self.assertEqual(accepted, {
                "model": str((model / "model.pt").resolve()), "manifest": str((model / "manifest.json").resolve()),
                "report": str((output / store.CONTINUOUS_RANKER_REPORT_FILENAME).resolve()), "score": str(score.resolve()),
                "model_sha256": build_file_manifest(model / "model.pt")["sha256"],
                "score_sha256": build_file_manifest(score)["sha256"], "selected_epoch": 3,
                "training_elapsed_sec": 0., "score_execution_start": "2021-01-01",
                "score_available_from": "2021-01-04", "score_available_through": "2021-01-04", "fold_count": None,
            })
            mutations = (
                ("model_sha", lambda m, r: m["model"].update(sha256="0" * 64)),
                ("score_sha", lambda m, r: m["research_outputs"]["oos_scores_gzip"].update(sha256="0" * 64)),
                ("seed", lambda m, r: r["training"].update(seed=202)),
                ("objective", lambda m, r: m.update(training_objective="wrong")),
                ("loss", lambda m, r: r["training"].update(loss="wrong")),
                ("version", lambda m, r: m["experiment_settings"].update(fit_contract_version=999)),
                ("status", lambda m, r: r.update(status="INVALID")),
                ("sop", lambda m, r: r["standard_model_sop"].update(evaluation_mode="rolling")),
            )
            for name, mutate in mutations:
                with self.subTest(name=name):
                    m, r = deepcopy(manifest), deepcopy(report)
                    mutate(m, r)
                    (model / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
                    (output / store.CONTINUOUS_RANKER_REPORT_FILENAME).write_text(json.dumps(r), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        validate()
            (model / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (output / store.CONTINUOUS_RANKER_REPORT_FILENAME).write_text(json.dumps(report), encoding="utf-8")
            with patch.object(source_owner, "_forward_source_provenance_issues", return_value=("stale FIT source",)):
                with self.assertRaisesRegex(ValueError, "lineage"):
                    validate()
            with patch.object(store, "_validate_current_forward_score_source_lineage", side_effect=ValueError("stale score lineage")):
                with self.assertRaisesRegex(ValueError, "lineage"):
                    validate()

    def test_binding_blocks_naked_paths_and_replacement(self):
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            identity, _, _, _, score = write_forward_fixture(root / "model", root / "output")
            args = dict(project_root=str(root), **identity, ticker="T0001", signal_date="2021-01-04",
                        score_path_override=str(score))
            with self.assertRaisesRegex(ValueError, "verified"):
                store.lookup_continuous_ranker_oos_candidate_score(**args)
            binding = source_owner.load_forward_score_binding(dataset="full",
                project_root=root, **identity, expected_seed=101,
                model_dir=root / "model", research_output_dir=root / "output")
            self.assertEqual(store.lookup_continuous_ranker_oos_candidate_score(**args, score_binding=binding)["score"], 0.4)
            (root / "model" / "model.pt").write_bytes(b"changed after admission")
            with self.assertRaisesRegex(ValueError, "changed"):
                store.lookup_continuous_ranker_oos_candidate_score(**args, score_binding=binding)

    def test_isolated_runtime_requires_binding_and_preserves_report_projection(self):
        from services.research.strategy_compare_runtime_contract import resolve_strategy_compare_runtime_contract
        from core.buy_sort import BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            identity, _, _, _, score = write_forward_fixture(root / "model", root / "output")
            binding = source_owner.load_forward_score_binding(dataset="full",
                project_root=root, **identity, expected_seed=101,
                model_dir=root / "model", research_output_dir=root / "output")
            args = dict(root=root, comparison_mode="score-ranking",
                        ranking_policy=BREAKOUT_QUALITY_RANKING_POLICY_RESOURCE_AWARE_CONTINUOUS_SCORE_NO_K_NO_R0,
                        ranking_options={}, optional_entry_filter_policy="current",
                        score_source=store.SCORE_SOURCE_CONTINUOUS_RANKER_OOS,
                        **identity, threshold=None, hard_filter_source=None,
                        continuous_score_path_override=str(score),
                        continuous_score_execution_start_override="2021-01-01")
            with self.assertRaisesRegex(ValueError, "verified"):
                resolve_strategy_compare_runtime_contract(**args)
            runtime = resolve_strategy_compare_runtime_contract(
                **args, continuous_score_bindings={str(score.resolve()): binding})
            self.assertEqual(runtime.continuous_score_override, {
                "score_path": str(score.resolve()), "execution_start": "2021-01-01",
                "available_from": "2021-01-04", "available_through": "2021-01-04"})
            self.assertNotIn("score_bindings", runtime.ranking_source)
            self.assertNotIn("score_bindings", runtime.ranking_options)
            args["continuous_score_execution_start_override"] = "2020-01-01"
            with self.assertRaisesRegex(ValueError, "execution_start"):
                resolve_strategy_compare_runtime_contract(
                    **args, continuous_score_bindings={str(score.resolve()): binding})


class ForwardPublicationTests(unittest.TestCase):
    def test_explicit_repair_rollback_cancels_only_its_new_forward_commits(self):
        from services.breakout_quality.fit_repair_transaction import preserved_fit_publications
        for phase in ("fit", "evaluation"):
            with self.subTest(phase=phase), TemporaryDirectory() as td:
                root = Path(td)
                model, output = root / "models" / "m", root / "outputs" / "o"
                model.mkdir(parents=True); output.mkdir(parents=True)
                (model / "model.pt").write_bytes(b"old")
                (output / "report.json").write_bytes(b"old evaluation")
                # A prior interrupted commit must finish BEFORE repair snapshots
                # its originals; rollback must preserve this prior validated FIT.
                with publication_owner.ForwardPublication(model, output) as prior:
                    self._write_fit(prior, content=b"prior validated FIT")
                    with patch.object(publication_owner, "atomic_replace_with_retry", side_effect=OSError("prior interruption")):
                        with self.assertRaises(OSError):
                            prior.commit_fit(checkpoint_validator=lambda p: None)
                def preflight():
                    self.assertEqual((model / "model.pt").read_bytes(), b"prior validated FIT")
                with self.assertRaisesRegex(OSError, "repair interruption"):
                    with preserved_fit_publications(
                        project_root=root, publication_roots=(model, output), journal_path=root / "repair.json",
                        repair_identity="test", forward_publication_roots=(model, output), locked_preflight=preflight,
                    ):
                        with publication_owner.ForwardPublication(model, output) as current:
                            self._write_fit(current, content=b"repair FIT")
                            if phase == "evaluation":
                                current.commit_fit(checkpoint_validator=lambda p: None)
                                (current.output_stage / "report.json").write_bytes(b"new report")
                                (current.model_stage / "manifest.json").write_bytes(b"new manifest")
                            with patch.object(publication_owner, "atomic_replace_with_retry", side_effect=OSError("repair interruption")):
                                if phase == "fit": current.commit_fit(checkpoint_validator=lambda p: None)
                                else: current.commit_evaluation(lambda *args: None)
                self.assertEqual((model / "model.pt").read_bytes(), b"prior validated FIT")
                self.assertEqual((output / "report.json").read_bytes(), b"old evaluation")
                with publication_owner.ForwardPublication(model, output) as resumed:
                    self.assertFalse(resumed._journal("fit").exists())
                    self.assertFalse(resumed._journal("evaluation").exists())

    def _write_fit(self, pub, content=b"new FIT"):
        (pub.model_stage / "model.pt").write_bytes(content)
        identity = {"training_settings": current_default_fitting_settings(), "seed": 101}
        write_fitted_model_contract(model_dir=pub.model_stage, fitting_identity=identity,
                                   selected_epoch=3, epoch_selection={}, final_refit_history=[],
                                   trainable_parameter_count=1, total_parameter_count=1)
        return identity

    def test_second_producer_conflicts_and_failed_evaluation_preserves_reusable_fit(self):
        with TemporaryDirectory() as td:
            model, output = Path(td) / "model", Path(td) / "output"
            model.mkdir(); output.mkdir()
            (model / "model.pt").write_bytes(b"old FIT")
            (model / "manifest.json").write_text('{"model":{"sha256":"old"}}')
            (output / "report.json").write_bytes(b"old evaluation")
            with publication_owner.ForwardPublication(model, output) as pub:
                with self.assertRaises(RuntimeError):
                    with publication_owner.ForwardPublication(model, output):
                        self.fail("another producer acquired the same namespace")
                identity = self._write_fit(pub)
                pub.commit_fit(expected_identity=identity)
                with self.assertRaisesRegex(ValueError, "evaluation failed"):
                    pub.commit_evaluation(lambda *args: (_ for _ in ()).throw(ValueError("evaluation failed")))
            self.assertEqual((model / "model.pt").read_bytes(), b"new FIT")
            self.assertEqual((output / "report.json").read_bytes(), b"old evaluation")
            contract, reason = load_reusable_fitted_model_contract(
                model_dir=model, expected_identity=identity, final_manifest_path=model / "manifest.json",
                report_path=output / "report.json")
            self.assertIsNotNone(contract, reason)
            self.assertEqual(contract.selected_epoch, 3)

    def test_repair_can_delegate_lease_but_another_thread_cannot_publish(self):
        from concurrent.futures import ThreadPoolExecutor
        with TemporaryDirectory() as td:
            model, output = Path(td) / "model", Path(td) / "output"
            def competing_producer():
                with publication_owner.ForwardPublication(model, output):
                    return "unexpected admission"
            with publication_owner.forward_publication_leases(model, output):
                with publication_owner.ForwardPublication(model, output) as child:
                    identity = self._write_fit(child)
                    child.commit_fit(expected_identity=identity)
                with ThreadPoolExecutor(max_workers=1) as pool:
                    with self.assertRaises(RuntimeError):
                        pool.submit(competing_producer).result(timeout=5)
            with publication_owner.ForwardPublication(model, output):
                self.assertEqual((model / "model.pt").read_bytes(), b"new FIT")

    def test_interrupted_commit_recovers_validated_fit_and_evaluation(self):
        for phase in ("fit", "evaluation"):
            with self.subTest(phase=phase), TemporaryDirectory() as td:
                model, output = Path(td) / "model", Path(td) / "output"
                with publication_owner.ForwardPublication(model, output) as pub:
                    identity = self._write_fit(pub)
                    if phase == "evaluation":
                        pub.commit_fit(expected_identity=identity)
                        (pub.output_stage / "report.json").write_bytes(b"new evaluation")
                        (pub.model_stage / "manifest.json").write_bytes(b"validated manifest")
                    original = publication_owner.atomic_replace_with_retry
                    calls = []
                    def interrupted(source, target):
                        calls.append(str(target))
                        if len(calls) == 2:
                            raise KeyboardInterrupt("simulated process interruption")
                        return original(source, target)
                    with patch.object(publication_owner, "atomic_replace_with_retry", side_effect=interrupted):
                        with self.assertRaises(KeyboardInterrupt):
                            if phase == "fit":
                                pub.commit_fit(expected_identity=identity)
                            else:
                                pub.commit_evaluation(lambda *args: None)
                    self.assertTrue(pub._journal(phase).exists())
                with publication_owner.ForwardPublication(model, output) as recovered:
                    self.assertFalse(recovered._journal(phase).exists())
                    self.assertEqual((model / "model.pt").read_bytes(), b"new FIT")
                    if phase == "evaluation":
                        self.assertEqual((output / "report.json").read_bytes(), b"new evaluation")
                        self.assertEqual((model / "manifest.json").read_bytes(), b"validated manifest")

    def test_recovery_rejects_wrong_phase_duplicates_and_unvalidated_journal(self):
        for fault in ("phase", "duplicate", "state"):
            with self.subTest(fault=fault), TemporaryDirectory() as td:
                model, output = Path(td) / "model", Path(td) / "output"
                with publication_owner.ForwardPublication(model, output) as pub:
                    model.mkdir(exist_ok=True); output.mkdir(exist_ok=True)
                    target = (output / "report.json") if fault == "phase" else model / "model.pt"
                    pending = target.with_name(".forward-commit-test")
                    pending.write_bytes(b"new")
                    entry = {"pending": str(pending), "target": str(target),
                             "sha256": build_file_manifest(pending)["sha256"]}
                    journal = {"version": 1, "phase": "fit", "state": "VALIDATED",
                               "entries": [entry, entry] if fault == "duplicate" else [entry]}
                    if fault == "state": journal["state"] = "WRITING"
                    pub._journal("fit").write_text(json.dumps(journal), encoding="utf-8")
                    with self.assertRaises(ValueError): pub.recover("fit")
                    self.assertFalse(target.exists())

    def test_stage_preserves_payload_bytes_and_unique_paths(self):
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            model, output = root / "model", root / "output"
            with publication_owner.ForwardPublication(model, output) as pub:
                identity, manifest, report, fit, _ = write_forward_fixture(pub.model_stage, pub.output_stage)
                expected = {str(path.relative_to(stage)): path.read_bytes()
                            for stage in (pub.model_stage, pub.output_stage) for path in stage.iterdir()}
                pub.commit_fit(expected_identity=fit)
                def validate(m, o):
                    source_owner.load_forward_score_binding(dataset="full", project_root=root, **identity, expected_seed=101,
                                                            model_dir=m, research_output_dir=o)
                pub.commit_evaluation(validate)
                for path in (*model.iterdir(), *output.iterdir()):
                    self.assertEqual(path.read_bytes(), expected[path.name])
                for path in (*model.glob("*.json"), *output.glob("*.json")):
                    self.assertNotIn("forward-stage-", path.read_text(encoding="utf-8"))
                first_stage = pub.model_stage
            with publication_owner.ForwardPublication(model, output) as second:
                self.assertNotEqual(first_stage, second.model_stage)

    def test_torch_checkpoint_basename_preserves_bytes(self):
        import torch
        from services.breakout_quality.train_daily_ranker import run as daily_run
        from services.breakout_quality.train_continuous_ranker import run as event_run
        self.assertTrue(hasattr(daily_run, "__wrapped__"))
        self.assertTrue(hasattr(event_run, "__wrapped__"))
        payload = {"model_state_dict": {"weight": torch.tensor([[1., 2.], [3., 4.]])},
                   "experiment_profile": "synthetic", "selected_epoch": 3}
        with TemporaryDirectory() as td:
            root = Path(td)
            old = root / "original" / "model.pt"
            old.parent.mkdir()
            torch.save(payload, old)
            with publication_owner.ForwardPublication(root / "model", root / "output") as pub:
                torch.save(payload, pub.model_stage / "model.pt")
                self.assertEqual(old.read_bytes(), (pub.model_stage / "model.pt").read_bytes())
                pub.commit_fit(checkpoint_validator=lambda p: self.assertTrue(
                    torch.equal(torch.load(p, weights_only=True)["model_state_dict"]["weight"],
                                payload["model_state_dict"]["weight"])))
                self.assertEqual(old.read_bytes(), (root / "model" / "model.pt").read_bytes())

    def test_actual_producer_wrapper_stages_args_and_reuses_fit_after_failure(self):
        settings = get_breakout_quality_model_research_settings()
        with TemporaryDirectory() as td, independent_upstream_evidence():
            root = Path(td)
            args = SimpleNamespace(filter_id=settings.filter_id, model_architecture=settings.model_architecture,
                                   experiment_profile=settings.experiment_profile, seed=101,
                                   model_output_dir=str(root / "model"), research_output_dir=str(root / "output"),
                                   reuse_fitted_model=False)
            original_args = vars(args).copy()
            trained = []
            fail_evaluation = [True]
            def producer(staged):
                from services.breakout_quality.train_continuous_ranker import resolve_training_output_paths
                paths, output = resolve_training_output_paths(staged)
                self.assertNotEqual(staged.model_output_dir, args.model_output_dir)
                identity = {"training_settings": current_default_fitting_settings(), "seed": 101}
                reusable, _ = load_reusable_fitted_model_contract(
                    model_dir=paths.model_dir, expected_identity=identity,
                    final_manifest_path=paths.manifest_path,
                    report_path=output / store.CONTINUOUS_RANKER_REPORT_FILENAME)
                if not staged.reuse_fitted_model or reusable is None:
                    trained.append(1)
                    write_forward_fixture(paths.model_dir, output)
                else:
                    # Populate evaluation fixture using real producer serializers,
                    # preserving the already admitted FIT bytes exactly.
                    fit_files = {name: (paths.model_dir / name).read_bytes()
                                 for name in ("model.pt", "fitted_model_manifest.json")}
                    write_forward_fixture(paths.model_dir, output)
                    for name, content in fit_files.items():
                        (paths.model_dir / name).write_bytes(content)
                publication_owner.commit_forward_fit(expected_identity=identity)
                if fail_evaluation[0]:
                    raise ValueError("synthetic evaluation failure")
                return 0
            with patch.dict(producer.__globals__, {"PROJECT_ROOT": root}):
                wrapped = publication_owner.forward_training_publication(producer)
                with self.assertRaisesRegex(ValueError, "evaluation failure"):
                    wrapped(args)
                self.assertEqual(vars(args), original_args)
                args.reuse_fitted_model = True
                fail_evaluation[0] = False
                self.assertEqual(wrapped(args), 0)
            self.assertEqual(len(trained), 1)
            for folder in (root / "model", root / "output"):
                for path in folder.glob("*.json"):
                    self.assertNotIn("forward-stage-", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
