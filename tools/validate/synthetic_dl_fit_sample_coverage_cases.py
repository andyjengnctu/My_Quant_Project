"""Focused capability tests for the reusable all-DL FIT sample Audit.

No canonical training, Target, Optimizer, test_suite or artifact producer runs.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.audit_policy import get_reusable_audit_definitions
from core.breakout_quality_policy import get_continuous_ranker_execution_recipe
from core.breakout_quality_registry import (
    TRAINING_LABEL_SCOPE_ALL, TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS, TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from core.research_report_contract import validate_approved_persistent_report_contracts
from core.breakout_quality_runtime import (
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH,
    CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
)
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME, LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.pit_group_selection import _fold_group_ids
from services.breakout_quality.fitting_data_dependencies import (
    FitDataDependencies, fold_source_dependency_issues,
)
from services.audit import dl_fit_sample_coverage as audit
from core.positive_netpnl_contract import POSITIVE_RANK_CALIBRATION
from filters.breakout_quality.positive_rank_calibration import calibration_enabled


class SampleCoverageTests(unittest.TestCase):
    @staticmethod
    def external_daily_profile():
        from core.breakout_quality_runtime import (
            CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE,
            CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT,
        )
        for _model, profile in audit._inventory():
            recipe = get_continuous_ranker_execution_recipe(profile)
            if (recipe.context_policy.source != CONTINUOUS_RANKER_CONTEXT_SOURCE_NONE
                and recipe.target_policy.materialization_mode
                == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT):
                return profile
        raise AssertionError("No registered daily-component external-context profile")
    def fake_bundle(self, *, daily=True):
        dates = pd.to_datetime([
            "2014-01-01", "2015-01-01", "2020-09-01", "2020-10-01",
            "2020-11-01", "2020-12-01", "2020-12-10", "2021-01-01",
        ])
        end = pd.to_datetime([
            "2014-03-01", "2015-02-01", "2020-12-01", "2020-11-01",
            "2021-01-10", None, "2020-12-18", "2021-02-01",
        ])
        valid = np.array([True, True, True, True, True, False, True, True])
        label = np.array([LABEL_PASS] * 6 + [LABEL_REJECT, LABEL_PASS])
        profile = SimpleNamespace(
            training_sample_scope=(TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                                   if daily else TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS),
            training_label_scope=(TRAINING_LABEL_SCOPE_ALL if daily else TRAINING_LABEL_SCOPE_PASS_ONLY),
            continuous_target_id="test_target",
            score_calibration=POSITIVE_RANK_CALIBRATION,
        )
        return SimpleNamespace(
            profile=profile,
            group_table=pd.DataFrame({
                "ticker": [f"T{i}" for i in range(8)],
                "date": dates,
                "label_eval_end_date": end,
                "label": label,
                "source_pos": np.arange(8),
            }),
            event_group_index=np.array([0, 0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64),
            raw_target=np.array([100., -30., 0., 50., -10., np.nan, -5., 7.], dtype=np.float32),
            target_valid=valid,
            summary={"policy": {"test": 1}, "dataset_storage_schema_version": 1,
                     "source_data_inventory": {"test": "same"},
                     "training_universe_start_date": "2014-01-01"},
            target_manifest={"schema_version": 1, "target_contract": {"target_kind": "not_real_path"},
                             "artifacts": {"source": "constant"}},
        )

    def fold(self, bundle):
        return _fold_group_ids(
            bundle, {"fold_id": "fold_001", "score_start": pd.Timestamp("2021-01-01"),
                     "score_end": pd.Timestamp("2021-01-31")}, validation_months=3,
        )

    def test_daily_partition_and_binary_maturity(self):
        bundle = self.fake_bundle(daily=True)
        ids = self.fold(bundle)
        cohort = audit._prepare_cohort(bundle)
        parts = audit._partitions(
            cohort["dates"].lt(pd.Timestamp("2021-01-01")).to_numpy(),
            cohort, score_start=pd.Timestamp("2021-01-01"), ids=ids,
        )
        self.assertEqual(parts["pre_score_groups"], 7)
        self.assertEqual(parts["missing_target"], 1)
        self.assertEqual(parts["scope_excluded"], 0)
        self.assertEqual(parts["not_matured"], 1)
        self.assertEqual(parts["final_refit"], 5)
        self.assertEqual(len(ids["train_ids"]), 2)
        self.assertEqual(len(ids["validation_ids"]), 2)
        positive, negative, rate = audit._positive_counts(
            cohort, parts["fit_mask"], has_positive_output=True,
        )
        self.assertEqual((positive, negative), (2, 3))
        self.assertAlmostEqual(rate, 40.)
        rows = audit._year_rows("MODEL", 1, "fold_001", cohort, {**parts, "pre_score_mask": cohort["dates"].lt("2021-01-01").to_numpy()}, has_positive_output=True)
        self.assertEqual(sum(row["final_refit"] for row in rows), 5)
        self.assertEqual(sum(row["missing_target"] for row in rows), 1)
        self.assertEqual(sum(row["event_rows"] for row in rows), 6)

    def test_event_scope_does_not_fake_negative(self):
        bundle = self.fake_bundle(daily=False)
        ids = self.fold(bundle)
        cohort = audit._prepare_cohort(bundle)
        parts = audit._partitions(
            cohort["dates"].lt("2021-01-01").to_numpy(), cohort,
            score_start=pd.Timestamp("2021-01-01"), ids=ids,
        )
        self.assertEqual(parts["scope_excluded"], 1)
        self.assertEqual(parts["not_matured"], 1)
        self.assertEqual(parts["missing_target"], 1)
        self.assertEqual(parts["final_refit"], 4)
        self.assertEqual(cohort["event_weights"][ids["final_ids"]].sum(), 5)
        self.assertIsNone(audit._positive_counts(cohort, parts["fit_mask"], has_positive_output=False)[2])

    def make_published_fold(self, root: Path, bundle, *, profile="test-profile"):
        base = root / "models" / "research" / "breakout_quality" / "model_robustness" / "rolling_oos" / "test-profile" / "seed_7"
        fold_dir = base / "folds" / "fold_001"
        fold_dir.mkdir(parents=True)
        ids = self.fold(bundle)
        cohort = audit._prepare_cohort(bundle)
        planned = {"validation_start": "2020-10-01", "validation_end": "2020-12-31",
                   "score_start": "2021-01-01", "score_end": "2021-01-31"}
        phase_ids = {"inner_train": ids["train_ids"], "validation": ids["validation_ids"],
                     "final_refit": ids["final_ids"], "score": ids["score_ids"]}
        observed = {name: audit._date_range(cohort["dates"], phase) for name, phase in phase_ids.items()}
        counts = {name: len(phase) for name, phase in phase_ids.items()}
        event_counts = {name: int(cohort["event_weights"][phase].sum()) for name, phase in phase_ids.items()}
        cutoff = str(cohort["maturity"].iloc[ids["final_ids"]].max().date())
        source = FitDataDependencies.from_bundle(bundle).rolling_source(
            fit_cutoff=pd.Timestamp(cutoff),
            daily_training_universe_start_date=(
                "2014-01-01" if bundle.profile.training_sample_scope
                == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS else None
            ),
        )
        manifest = {
            "fold_id": "fold_001", "seed": 7, "experiment_profile": profile,
            "filter_id": "test_filter", "model_architecture": "test_arch",
            "training_sample_scope": bundle.profile.training_sample_scope,
            "planned_periods": planned, "observed_periods": observed,
            "group_counts": counts, "event_row_counts": event_counts,
            "model_information_cutoff": cutoff, "training_settings": {},
            "source_contract": source,
        }
        fold_path = fold_dir / "manifest.json"
        fold_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        paths = {"fold_manifest": fold_path,
                 "checkpoint": fold_dir / DEFAULT_MODEL_FILENAME,
                 "scores": fold_dir / "scores.csv",
                 "validation_scores": fold_dir / "validation_scores.csv"}
        for key, path in paths.items():
            if key != "fold_manifest":
                path.write_bytes(f"synthetic_{key}".encode())
        records = {key: build_file_manifest(path) for key, path in paths.items()}
        aggregate = {"folds": [{
            "fold_id": "fold_001", "planned_periods": planned, "observed_periods": observed,
            "group_counts": counts, "event_row_counts": event_counts,
            "model_information_cutoff": cutoff,
            "fold_manifest": records["fold_manifest"],
            "artifacts": {key: records[key] for key in ("checkpoint", "scores", "validation_scores")},
        }]}
        (base / "selection_point_in_time_manifest.json").write_text(json.dumps(aggregate), encoding="utf-8")
        return base, manifest, paths

    def test_published_source_fold_sha_and_mismatch_detection(self):
        bundle = self.fake_bundle()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, manifest, paths = self.make_published_fold(root, bundle)
            args = {"filter_id": "test_filter", "model_architecture": "test_arch"}
            fake_settings = SimpleNamespace(**args)
            fake_recipe = SimpleNamespace(
                training_policy=SimpleNamespace(target_builder="scalar_pairwise"),
                target_policy=SimpleNamespace(materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH),
                context_policy=SimpleNamespace(source="none"),
            )
            with (patch.object(audit, "_source_path", return_value=(base, "test_filter", "test_arch")),
                  patch.object(audit, "get_breakout_quality_workflow_settings", return_value=fake_settings),
                  patch.object(audit, "resolve_feature_price_source", return_value=None),
                  patch.object(audit, "resolve_trade_count_feature_source", return_value=None),
                  patch.object(audit, "loader_kwargs_for_trade_path", return_value={}),
                  patch.object(audit, "load_profile_continuous_ranker_data", return_value=bundle),
                  patch.object(audit, "get_continuous_ranker_execution_recipe", return_value=fake_recipe)):
                result = audit._audit_seed(root, model="MODEL", profile="test-profile", seed=7)
                self.assertEqual(result["status"], "VERIFIED")
                self.assertEqual(result["seed_coverage"]["final_refit"], 5)
                self.assertEqual(result["seed_coverage"]["positive_rate"], 40.0)
                self.assertEqual(result["scope"], "daily / trade-path")
                # Authentic signed Fold source matching is mandatory.
                bad = dict(manifest)
                bad["source_contract"] = {**manifest["source_contract"], "target_schema_version": 99}
                paths["fold_manifest"].write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "雜湊不一致"):
                    audit._audit_seed(root, model="MODEL", profile="test-profile", seed=7)
                # A maliciously re-signed but source-stale Fold remains rejected.
                aggregate_path = base / "selection_point_in_time_manifest.json"
                agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
                agg["folds"][0]["fold_manifest"] = build_file_manifest(paths["fold_manifest"])
                aggregate_path.write_text(json.dumps(agg), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    audit._audit_seed(root, model="MODEL", profile="test-profile", seed=7)

    def test_partial_report_publish_and_model_registry_driven(self):
        bundle = self.fake_bundle()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, _, _ = self.make_published_fold(root, bundle)
            signed_path = base / "selection_point_in_time_manifest.json"
            signed_before = signed_path.read_bytes()
            fake_settings = SimpleNamespace(filter_id="test_filter", model_architecture="test_arch")
            fake_recipe = SimpleNamespace(
                training_policy=SimpleNamespace(target_builder="scalar_pairwise"),
                target_policy=SimpleNamespace(materialization_mode=CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH),
                context_policy=SimpleNamespace(source="none"),
            )
            definitions = get_reusable_audit_definitions("breakout_quality")
            definition = next(d for d in definitions if d.audit_type == audit.SUPPORTED_AUDIT_TYPE)
            def path_for_seed(_root, _profile, seed):
                return ((base if seed == 7 else base.parent / "seed_8"), "test_filter", "test_arch")
            with (patch.object(audit, "_inventory", return_value=(("MODEL-A", "test-profile"),)),
                  patch.object(audit, "_seeds", return_value=(7, 8)),
                  patch.object(audit, "_source_path", side_effect=path_for_seed),
                  patch.object(audit, "get_breakout_quality_workflow_settings", return_value=fake_settings),
                  patch.object(audit, "resolve_feature_price_source", return_value=None) as price_source,
                  patch.object(audit, "resolve_trade_count_feature_source", return_value=None),
                  patch.object(audit, "loader_kwargs_for_trade_path", return_value={}) as trade_loader,
                  patch.object(audit, "load_profile_continuous_ranker_data", return_value=bundle),
                  patch.object(audit, "get_continuous_ranker_execution_recipe", return_value=fake_recipe)):
                saved = audit.run_formal_audit(definition, project_root=root, quiet=True)
                self.assertFalse(price_source.call_args.kwargs["create"])
                self.assertFalse(trade_loader.call_args.kwargs["authorized_build"])
            self.assertEqual(signed_path.read_bytes(), signed_before)
            self.assertEqual(saved["status"], "PARTIAL")
            self.assertEqual(saved["verified_seed_count"], 1)
            self.assertEqual(saved["expected_seed_count"], 2)
            self.assertEqual(saved["models"][0]["status"], "PARTIAL")
            self.assertEqual(len(saved["folds"]), 1)
            self.assertEqual(len(saved["blocked"]), 1)
            self.assertEqual(saved["blocked"][0]["seed"], 8)
            md = root / "outputs" / "audit" / definition.output_subdir / "latest" / "audit.md"
            self.assertTrue(md.is_file())
            report_text = md.read_text(encoding="utf-8")
            self.assertIn("MODEL-A", report_text)
            self.assertIn("FileNotFoundError", json.dumps(saved["blocked"], ensure_ascii=False))
            self.assertIn("Yearly each-Fold", report_text)

    def test_every_registered_dl_recipe_uses_canonical_training_policy(self):
        """Execute the real profile contract through the Audit, not a mirrored mock.

        This catches an accidental objective_policy.target_builder lookup even if
        synthetic stand-ins are later changed to match the same wrong field.
        """
        inventory = audit._inventory()
        self.assertTrue(inventory)
        for model, profile in inventory:
            with self.subTest(model=model, profile=profile):
                recipe = get_continuous_ranker_execution_recipe(profile)
                from core.breakout_quality_registry import get_breakout_quality_experiment_profile
                registered_profile = get_breakout_quality_experiment_profile(profile)
                is_positive = calibration_enabled(registered_profile)
                if is_positive:
                    self.assertIsNone(recipe.training_policy.categorical_class_count)
                    self.assertTrue(recipe.training_policy.uses_pairwise_loss)
                daily = recipe.training_sample_scope == TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
                bundle = self.fake_bundle(daily=daily)
                bundle.profile.score_calibration = registered_profile.score_calibration
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    base, _manifest, _paths = self.make_published_fold(root, bundle, profile=profile)
                    fake_settings = SimpleNamespace(filter_id="test_filter", model_architecture="test_arch")
                    # The persisted synthetic source is a fixture, but the
                    # execution recipe is resolved from the *real* registry.
                    with (patch.object(audit, "_source_path", return_value=(base, "test_filter", "test_arch")),
                          patch.object(audit, "get_breakout_quality_workflow_settings", return_value=fake_settings),
                          patch.object(audit, "resolve_feature_price_source", return_value=None),
                          patch.object(audit, "resolve_trade_count_feature_source", return_value=None),
                          patch.object(audit, "loader_kwargs_for_trade_path", return_value={}),
                          patch.object(audit, "load_profile_continuous_ranker_data", return_value=bundle),
                          # This test exercises genuine registry Recipe shape;
                          # external historical lineage is covered by the
                          # dedicated positive/negative proof tests below.
                          patch.object(audit, "verified_published_context_fit_proof", return_value=(True, "test fixture"))):
                        actual = audit._audit_seed(root, model=model, profile=profile, seed=7)
                self.assertEqual(actual["status"], "VERIFIED")
                expected_positive_rate = None
                if is_positive:
                    fitted_raw = bundle.raw_target[self.fold(bundle)["final_ids"]]
                    expected_positive_rate = float(100 * (fitted_raw > 0).sum() / len(fitted_raw))
                self.assertEqual(actual["seed_coverage"]["positive_rate"], expected_positive_rate)
                self.assertEqual(actual["label_materialization"], recipe.target_policy.materialization_mode)

    def test_daily_context_source_drift_uses_formal_scope_and_signed_context(self):
        """Context-dependent Daily FIT may ignore *nonconsumed* cache drift only.

        A full registered profile and owner-level source checker are used.  The
        historical context proof itself is mocked here because the synthetic
        fixture does not contain a real Stage-1 model or checkpoint lineage.
        """
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        from filters.breakout_quality.model_source_provenance import source_inventory_matches_current

        profile = self.external_daily_profile()
        recipe = get_continuous_ranker_execution_recipe(profile)
        self.assertNotEqual(recipe.context_policy.source, "none")
        self.assertEqual(recipe.target_policy.materialization_mode, CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_COMPONENT)
        dataset = resolve_dataset_paths(Path(".")).artifact_paths()
        def records(value):
            return {key: {"filename": path.name, "sha256": value * 64, "size_bytes": 1}
                    for key, path in dataset.items()}
        common = {"source_data_inventory": {"same_csv": "identity"},
                  "target_contract": {"target_kind": "pure_mfe"},
                  "target_artifacts": {"source": "same_target"}}
        old_source = {**common, "dataset_artifacts": records("a")}
        new_source = {**common, "dataset_artifacts": records("b")}
        candidate = {"experiment_profile": profile,
                     "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
                     "source_contract": old_source}
        expected = {"experiment_profile": profile,
                    "training_sample_scope": TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
                    "source_contract": new_source}
        # This test uses complete registry-resolved metadata. No AH special-case.
        self.assertFalse(fold_source_dependency_issues(
            candidate, expected, project_root=Path("."),
            inventory_matcher=source_inventory_matches_current,
            external_context_verified=True,
        ))
        blocked = fold_source_dependency_issues(
            candidate, expected, project_root=Path("."),
            inventory_matcher=source_inventory_matches_current,
            external_context_verified=False,
        )
        self.assertTrue(any("external context" in msg for msg in blocked))
        wrong_scope = {**expected, "training_sample_scope": TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS}
        self.assertTrue(fold_source_dependency_issues(
            candidate, wrong_scope, project_root=Path("."),
            inventory_matcher=source_inventory_matches_current,
            external_context_verified=True,
        ))
        changed_target = {**expected, "source_contract": {**new_source,
                          "target_contract": {"target_kind": "different_target"}}}
        self.assertTrue(fold_source_dependency_issues(
            candidate, changed_target, project_root=Path("."),
            inventory_matcher=source_inventory_matches_current,
            external_context_verified=True,
        ))

    def test_external_context_attestation_accepts_only_eligible_dataset_drift(self):
        """Integration: all 6 source checks are bound to original Fold provenance."""
        from filters.breakout_quality.dataset_store import resolve_dataset_paths
        profile = self.external_daily_profile()
        bundle = self.fake_bundle()
        paths = resolve_dataset_paths(Path(".")).artifact_paths()
        def records(value):
            return {key: {"filename": path.name, "sha256": value * 64, "size_bytes": 1}
                    for key, path in paths.items()}
        bundle.summary["dataset_artifacts"] = records("b")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, manifest, files = self.make_published_fold(root, bundle, profile=profile)
            old = {**manifest, "source_contract": {
                **manifest["source_contract"], "dataset_artifacts": records("a")}}
            files["fold_manifest"].write_text(json.dumps(old), encoding="utf-8")
            aggregate_path = base / "selection_point_in_time_manifest.json"
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            aggregate["folds"][0]["fold_manifest"] = build_file_manifest(files["fold_manifest"])
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
            settings = SimpleNamespace(filter_id="test_filter", model_architecture="test_arch")
            with (patch.object(audit, "_source_path", return_value=(base, "test_filter", "test_arch")),
                  patch.object(audit, "get_breakout_quality_workflow_settings", return_value=settings),
                  patch.object(audit, "resolve_feature_price_source", return_value=None),
                  patch.object(audit, "resolve_trade_count_feature_source", return_value=None),
                  patch.object(audit, "loader_kwargs_for_trade_path", return_value={}),
                  patch.object(audit, "load_profile_continuous_ranker_data", return_value=bundle),
                  patch.object(audit, "verified_published_context_fit_proof", return_value=(True, "proven")) as proof):
                actual = audit._audit_seed(root, model="MODEL", profile=profile, seed=7)
                self.assertEqual(actual["status"], "VERIFIED")
                self.assertEqual(proof.call_count, 1)
                bad = {**old, "source_contract": {**old["source_contract"],
                       "target_contract": {"target_kind": "different_target"}}}
                files["fold_manifest"].write_text(json.dumps(bad), encoding="utf-8")
                aggregate["folds"][0]["fold_manifest"] = build_file_manifest(files["fold_manifest"])
                aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source mismatch"):
                    audit._audit_seed(root, model="MODEL", profile=profile, seed=7)

    def test_external_context_is_required_even_without_cache_drift(self):
        """An AH-like context cannot be marked VERIFIED from matching hashes."""
        profile = self.external_daily_profile()
        bundle = self.fake_bundle()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base, _manifest, paths = self.make_published_fold(root, bundle, profile=profile)
            settings = SimpleNamespace(filter_id="test_filter", model_architecture="test_arch")
            with (patch.object(audit, "_source_path", return_value=(base, "test_filter", "test_arch")),
                  patch.object(audit, "get_breakout_quality_workflow_settings", return_value=settings),
                  patch.object(audit, "resolve_feature_price_source", return_value=None),
                  patch.object(audit, "resolve_trade_count_feature_source", return_value=None),
                  patch.object(audit, "loader_kwargs_for_trade_path", return_value={}),
                  patch.object(audit, "load_profile_continuous_ranker_data", return_value=bundle)):
                # Real owner verification must fail on a fixture without
                # authentic Stage-1 Context and its original FIT binding.
                with self.assertRaisesRegex(ValueError, "original FIT context proof blocked"):
                    audit._audit_seed(root, model="MODEL", profile=profile, seed=7)
                # Isolate audit plumbing and inspect proof arguments.
                with patch.object(audit, "verified_published_context_fit_proof", return_value=(False, "original FIT proof missing")) as proof:
                    with self.assertRaisesRegex(ValueError, "original FIT context proof blocked"):
                        audit._audit_seed(root, model="MODEL", profile=profile, seed=7)
                    self.assertEqual(proof.call_count, 1)
                    self.assertEqual(proof.call_args.kwargs["manifest_path"], paths["fold_manifest"])
                    self.assertIs(proof.call_args.kwargs["bundle"], bundle)
                    self.assertEqual(proof.call_args.kwargs["project_root"], root)

    def test_registry_and_report_approval(self):
        self.assertEqual(validate_approved_persistent_report_contracts(), ())
        definitions = get_reusable_audit_definitions("breakout_quality")
        match = [d for d in definitions if d.audit_type == audit.SUPPORTED_AUDIT_TYPE]
        self.assertEqual(len(match), 1)
        with tempfile.TemporaryDirectory() as temp:
            state = audit.collect_status(match[0], project_root=Path(temp))
        self.assertEqual(state["status"], "BLOCKED")
        self.assertIn("不建立前置工件", state["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
