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
from core.breakout_quality_registry import (
    TRAINING_LABEL_SCOPE_ALL, TRAINING_LABEL_SCOPE_PASS_ONLY,
    TRAINING_SAMPLE_SCOPE_BREAKOUT_EVENT_GROUPS, TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS,
)
from core.research_report_contract import validate_approved_persistent_report_contracts
from filters.breakout_quality.artifacts import build_file_manifest
from filters.breakout_quality.contract import DEFAULT_MODEL_FILENAME, LABEL_PASS, LABEL_REJECT
from filters.breakout_quality.pit_group_selection import _fold_group_ids
from services.breakout_quality.fitting_data_dependencies import FitDataDependencies
from services.audit import dl_fit_sample_coverage as audit


class SampleCoverageTests(unittest.TestCase):
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
            cohort, parts["fit_mask"], is_positive_binary=True,
        )
        self.assertEqual((positive, negative), (2, 3))
        self.assertAlmostEqual(rate, 40.)
        rows = audit._year_rows("MODEL", 1, "fold_001", cohort, {**parts, "pre_score_mask": cohort["dates"].lt("2021-01-01").to_numpy()}, is_positive_binary=True)
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
        self.assertIsNone(audit._positive_counts(cohort, parts["fit_mask"], is_positive_binary=False)[2])

    def make_published_fold(self, root: Path, bundle):
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
        source = FitDataDependencies.from_bundle(bundle).rolling_source(
            fit_cutoff=pd.Timestamp("2020-12-18"),
            daily_training_universe_start_date="2014-01-01",
        )
        manifest = {
            "fold_id": "fold_001", "seed": 7, "experiment_profile": "test-profile",
            "filter_id": "test_filter", "model_architecture": "test_arch",
            "planned_periods": planned, "observed_periods": observed,
            "group_counts": counts, "event_row_counts": event_counts,
            "model_information_cutoff": "2020-12-18", "training_settings": {},
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
            "model_information_cutoff": "2020-12-18",
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
                objective_policy=SimpleNamespace(target_builder="binary_positive_from_raw"),
                target_policy=SimpleNamespace(materialization_mode="daily_trade_path"),
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
                objective_policy=SimpleNamespace(target_builder="binary_positive_from_raw"),
                target_policy=SimpleNamespace(materialization_mode="daily_trade_path"),
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
