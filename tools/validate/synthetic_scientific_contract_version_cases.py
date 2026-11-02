"""Synthetic FIT/Replay version semantics: rebuild affected publications only.

Uses real canonical profiles and Strategy Compare sources but tiny fake manifests;
never invokes formal Research training, Replay, or the local Test Suite.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
    DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE,
    get_breakout_quality_experiment_profile,
)
from core.strategy_compare_policy import get_strategy_comparison_settings
from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from filters.breakout_quality.strategy_compare_pit_contract import (
    load_validated_selection_pit_strategy_compare_contract,
)
from services.breakout_quality.fitted_model_artifacts import (
    published_fit_contract_version_state,
)
from services.breakout_quality.fit_repair_transaction import (
    superseded_fit_publication_guard,
)
from services.research import breakout_quality_application as application
from services.research import strategy_compare_training


CQ = DAILY_UNIVERSAL_T40_PRETRAINED_NETPNL_POSITIVE_BINARY_PROFILE
CP = DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE
FOLD = "fold_20210101_20211231"


def _manifest_settings(profile_name, *, legacy=False):
    settings = get_breakout_quality_experiment_profile(profile_name).as_manifest_payload()
    if legacy:
        settings.pop("fit_contract_version", None)
        pretrain = settings.get("encoder_pretraining")
        if pretrain is not None:
            pretrain["name"] = "daily_t40_regression_encoder_v1"
            pretrain["epochs"] = 100
            pretrain["sampling_mode"] = "ticker_balanced_one_window_per_ticker_per_epoch"
            pretrain.pop("validation_months", None)
            pretrain.pop("early_stopping_patience", None)
    return settings


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_rolling(path: Path, profile_name: str, *, legacy=False):
    _write_json(path / "selection_point_in_time_manifest.json", {
        "experiment_profile": profile_name,
        "folds": [{"fold_id": FOLD}],
    })
    _write_json(path / "folds" / FOLD / "manifest.json", {
        "experiment_profile": profile_name,
        "experiment_settings": _manifest_settings(profile_name, legacy=legacy),
    })


class ScientificVersionCases(TestCase):
    def test_single_registry_owner_publishes_only_nondefault_fit_version(self):
        cp = get_breakout_quality_experiment_profile(CP)
        cq = get_breakout_quality_experiment_profile(CQ)
        self.assertNotIn("fit_contract_version", cp.as_manifest_payload())
        self.assertEqual(cq.as_manifest_payload()["fit_contract_version"], 2)
        self.assertEqual(cq.encoder_pretraining_profile, get_breakout_quality_experiment_profile(CQ).encoder_pretraining_profile)

    def test_complete_model_staleness_and_reuse_after_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            _write_rolling(path, CQ, legacy=True)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "UPGRADE_REQUIRED")
            fake_settings = SimpleNamespace(experiment_profile=CQ)
            self.assertTrue(application._version_rebuild_required(
                fake_settings, rolling=True, publication_dir=path))
            _write_rolling(path, CQ)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "CURRENT")
            self.assertFalse(application._version_rebuild_required(
                fake_settings, rolling=True, publication_dir=path))
            # A forgotten version bump is a contract failure, not authorization to rebuild.
            fold = path / "folds" / FOLD / "manifest.json"
            modified = json.loads(fold.read_text(encoding="utf-8"))
            modified["experiment_settings"]["encoder_pretraining"]["epochs"] += 1
            _write_json(fold, modified)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "BLOCKED")
            with self.assertRaisesRegex(RuntimeError, "FIT contract unverified"):
                application._version_rebuild_required(fake_settings, rolling=True, publication_dir=path)

    def test_unaffected_cp_v1_and_older_v1_artifacts_remain_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp"
            _write_rolling(path, CP)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CP, publication_dir=path, rolling=True)[0], "CURRENT")
            forward = Path(tmp) / "cp_forward"
            _write_json(forward / "manifest.json", {
                "experiment_profile": CP, "experiment_settings": _manifest_settings(CP),
            })
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CP, publication_dir=forward, rolling=False)[0], "CURRENT")

    def test_crash_after_rename_still_requires_full_version_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            _write_rolling(path, CQ, legacy=True)
            backup = path.with_name("." + path.name + ".superseded_fit_backup")
            path.rename(backup)  # synthetic forced process termination
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "UPGRADE_REQUIRED")
            with superseded_fit_publication_guard(path):
                self.assertFalse(path.exists())
                _write_rolling(path, CQ)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "CURRENT")
            self.assertFalse(backup.exists())

    def test_corrupt_or_partial_source_does_not_authorize_version_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_json(path / "selection_point_in_time_manifest.json", {
                "experiment_profile": CQ, "folds": [{"fold_id": FOLD}],
            })
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "BLOCKED")

    def test_unreadable_existing_manifest_cannot_look_like_missing_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            path.mkdir()
            (path / "selection_point_in_time_manifest.json").write_text("{broken")
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=path, rolling=True)[0], "BLOCKED")
            forward = Path(tmp) / "forward"
            forward.mkdir()
            (forward / "manifest.json").write_text("{broken")
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=forward, rolling=False)[0], "BLOCKED")

    def test_strategy_arms_inherit_current_model_version_without_bumping_replay_rules(self):
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        for old, new in (("C125", "C127"), ("C126", "C128")):
            old_arm = settings.arms[old]
            new_arm = settings.arms[new]
            self.assertEqual(old_arm.replay_contract_version, 1)
            self.assertEqual(new_arm.replay_contract_version, 1)
            self.assertNotIn("replay_contract_version", new_arm.as_dict())
            v2 = replace(new_arm, replay_contract_version=2)
            self.assertEqual(v2.as_dict()["replay_contract_version"], 2)
            self.assertNotEqual(v2.as_dict(), new_arm.as_dict())

    def test_strategy_consumer_rejects_superseded_auxiliary_dl_source(self):
        source = get_strategy_comparison_settings(
            STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        ).dl_sources["CONT13CQ_ROLL"]
        workflow = SimpleNamespace(point_in_time_fold_months=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_rolling(path, CQ, legacy=True)
            record = SimpleNamespace(seed=42, manifest={
                "fold_months": 12,
                "single_score_block": bool(source.point_in_time_single_score_block),
                "folds": [{"fold_id": FOLD, "selected_epoch": 1}],
            })
            with patch('filters.breakout_quality.strategy_compare_pit_contract.load_selection_point_in_time_ranking_contract', return_value=record), patch('filters.breakout_quality.strategy_compare_pit_contract.validate_selection_pit_fold_anchor_identity'), patch('filters.breakout_quality.strategy_compare_pit_contract.validate_selection_pit_score_period'):
                with self.assertRaisesRegex(ValueError, "FIT identity not current"):
                    load_validated_selection_pit_strategy_compare_contract(
                        root=path, source=source, workflow=workflow, seed=42,
                        point_in_time_dir_override=path,
                    )
                _write_rolling(path, CQ)
                self.assertIs(load_validated_selection_pit_strategy_compare_contract(
                    root=path, source=source, workflow=workflow, seed=42,
                    point_in_time_dir_override=path,
                ), record)

    def test_one_canonical_publication_no_unbounded_archived_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "model"
            path.mkdir()
            (path / "checkpoint").write_text("v1")
            with superseded_fit_publication_guard(path):
                self.assertFalse(path.exists())
                path.mkdir()
                (path / "checkpoint").write_text("v2")
            self.assertEqual((path / "checkpoint").read_text(), "v2")
            self.assertFalse((root / ".model.superseded_fit_backup").exists())
            with superseded_fit_publication_guard(path):
                path.mkdir()
                (path / "checkpoint").write_text("v3")
            self.assertEqual((path / "checkpoint").read_text(), "v3")
            self.assertEqual([p.name for p in root.iterdir()], ["model"])
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with superseded_fit_publication_guard(path):
                    path.mkdir()
                    (path / "checkpoint").write_text("invalid")
                    raise RuntimeError("failed")
            self.assertEqual((path / "checkpoint").read_text(), "v3")
            self.assertFalse((root / ".model.superseded_fit_backup").exists())

    def test_multiroot_failed_setup_restores_previous_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "model"
            invalid = root / "not_a_dir"
            valid.mkdir()
            (valid / "checkpoint").write_text("untouched")
            invalid.write_text("file")
            with self.assertRaisesRegex(RuntimeError, "non-directory"):
                with superseded_fit_publication_guard(valid, invalid):
                    self.fail("unsafe setup must not begin rebuild")
            self.assertEqual((valid / "checkpoint").read_text(), "untouched")
            self.assertFalse((root / ".model.superseded_fit_backup").exists())

    def test_strategy_compare_stale_source_uses_no_resume_or_previous_checkpoint_cache(self):
        source = get_strategy_comparison_settings(
            STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        ).dl_sources["CONT13CQ_ROLL"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "published"
            _write_rolling(target, CQ, legacy=True)
            (target / "old_checkpoint").write_text("legacy model")
            captured = {}
            def fake_builder(**kwargs):
                captured.update(kwargs)
                return ["python", "fake"], [], {"score_source": kwargs["source"].score_source,
                    "score_start_date": "2021-01-01", "score_end_date": "2026-03-02"}
            def fake_train(**kwargs):
                self.assertFalse(target.exists())
                _write_rolling(target, CQ)
                return {"elapsed_sec": 0.0}
            with patch.object(strategy_compare_training, 'build_strategy_compare_trainer_command', side_effect=fake_builder), patch.object(strategy_compare_training, 'run_logged_training_process', side_effect=fake_train), patch.object(strategy_compare_training, 'validate_strategy_compare_training_artifacts', return_value={"ready":True}), patch('services.breakout_quality.point_in_time_audit.audit_selection_point_in_time_scores', return_value=0):
                result = strategy_compare_training.run_strategy_compare_training_unit(
                    project_root=root, source=source,
                    workflow=SimpleNamespace(point_in_time_fold_months=12), seed=42,
                    model_dir=target, research_dir=None,
                    log_path=root / "mock.log", comparison_start="2021-01-01",
                    comparison_end="2026-03-02", point_in_time_dir_override=target,
                    checkpoint_cache_root=root / "shared_cache", fit_source_pit_dirs=(root / "old",),
                )
            self.assertFalse(captured["resume"])
            self.assertIsNone(captured["checkpoint_cache_root"])
            self.assertEqual(captured["fit_source_pit_dirs"], ())
            self.assertTrue(result["artifacts"]["ready"])
            self.assertFalse((target / "old_checkpoint").exists())
            self.assertFalse((root / ".published.superseded_fit_backup").exists())
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=CQ, publication_dir=target, rolling=True)[0], "CURRENT")


if __name__ == "__main__":
    main()
