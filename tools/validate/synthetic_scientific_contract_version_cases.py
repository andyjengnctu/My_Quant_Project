"""Synthetic FIT/Replay version semantics: rebuild affected publications only.

Uses real canonical profiles and Strategy Compare sources but tiny fake manifests;
never invokes formal Research training, Replay, or the local Test Suite.
"""
from __future__ import annotations

import json
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch
from core import breakout_quality_registry as model_registry

from core.breakout_quality_registry import (
    DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE,
    DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE,
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


# Synthetic publication versions are built locally; no real FIT artifact is read.
VERSIONED_PROFILE = DAILY_UNIVERSAL_NETPNL_POSITIVE_BINARY_PROFILE
UNAFFECTED_PROFILE = DAILY_UNIVERSAL_FORMAL_MINTRADE_FULLCYCLE_NETPNL_PAIRWISE_PROFILE
FOLD = "fold_20210101_20211231"


def _manifest_settings(profile_name, *, legacy=False):
    settings = get_breakout_quality_experiment_profile(profile_name).as_manifest_payload()
    if legacy:
        settings.pop("fit_contract_version", None)
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
    def setUp(self):
        # The test must never change the persisted owner or any physical model FIT.
        original = get_breakout_quality_experiment_profile(VERSIONED_PROFILE)
        self.version_override = patch.dict(model_registry._EXPERIMENT_PROFILES, {
            VERSIONED_PROFILE: replace(original, fit_contract_version=2),
        })
        self.version_override.start()
        self.addCleanup(self.version_override.stop)

    def test_synthetic_upgrade_only_sets_version_in_local_owner(self):
        current = get_breakout_quality_experiment_profile(VERSIONED_PROFILE)
        other = get_breakout_quality_experiment_profile(UNAFFECTED_PROFILE)
        self.assertEqual(current.as_manifest_payload()["fit_contract_version"], 2)
        self.assertNotIn("fit_contract_version", other.as_manifest_payload())

    def test_complete_model_staleness_and_reuse_after_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            _write_rolling(path, VERSIONED_PROFILE, legacy=True)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "UPGRADE_REQUIRED")
            fake_settings = SimpleNamespace(experiment_profile=VERSIONED_PROFILE)
            self.assertTrue(application._version_rebuild_required(
                fake_settings, rolling=True, publication_dir=path))
            _write_rolling(path, VERSIONED_PROFILE)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "CURRENT")
            self.assertFalse(application._version_rebuild_required(
                fake_settings, rolling=True, publication_dir=path))
            # A forgotten version bump is a contract failure, not authorization to rebuild.
            fold = path / "folds" / FOLD / "manifest.json"
            modified = json.loads(fold.read_text(encoding="utf-8"))
            modified["experiment_settings"]["learning_rate"] = 0.12345
            _write_json(fold, modified)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "BLOCKED")
            with self.assertRaisesRegex(RuntimeError, "FIT contract unverified"):
                application._version_rebuild_required(fake_settings, rolling=True, publication_dir=path)

    def test_unaffected_model_v1_and_older_v1_artifacts_remain_reusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp"
            _write_rolling(path, UNAFFECTED_PROFILE)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=UNAFFECTED_PROFILE, publication_dir=path, rolling=True)[0], "CURRENT")
            forward = Path(tmp) / "other_forward"
            _write_json(forward / "manifest.json", {
                "experiment_profile": UNAFFECTED_PROFILE, "experiment_settings": _manifest_settings(UNAFFECTED_PROFILE),
            })
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=UNAFFECTED_PROFILE, publication_dir=forward, rolling=False)[0], "CURRENT")

    def test_crash_after_rename_still_requires_full_version_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            _write_rolling(path, VERSIONED_PROFILE, legacy=True)
            backup = path.with_name("." + path.name + ".superseded_fit_backup")
            path.rename(backup)  # synthetic forced process termination
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "UPGRADE_REQUIRED")
            with superseded_fit_publication_guard(path):
                self.assertFalse(path.exists())
                _write_rolling(path, VERSIONED_PROFILE)
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "CURRENT")
            self.assertFalse(backup.exists())

    def test_corrupt_or_partial_source_does_not_authorize_version_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_json(path / "selection_point_in_time_manifest.json", {
                "experiment_profile": VERSIONED_PROFILE, "folds": [{"fold_id": FOLD}],
            })
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "BLOCKED")

    def test_unreadable_existing_manifest_cannot_look_like_missing_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rolling"
            path.mkdir()
            (path / "selection_point_in_time_manifest.json").write_text("{broken")
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True)[0], "BLOCKED")
            forward = Path(tmp) / "forward"
            forward.mkdir()
            (forward / "manifest.json").write_text("{broken")
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=forward, rolling=False)[0], "BLOCKED")

    def test_strategy_arms_inherit_current_model_version_without_bumping_replay_rules(self):
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        for old, new in (("C123", "C125"), ("C124", "C126")):
            old_arm = settings.arms[old]
            new_arm = settings.arms[new]
            self.assertEqual(old_arm.replay_contract_version, 1)
            from core.strategy_compare_registry import STRATEGY_COMPARE_ARMS
            declared = STRATEGY_COMPARE_ARMS[new].get("replay_contract_version", 1)
            self.assertEqual(new_arm.replay_contract_version, declared)
            next_version = replace(new_arm, replay_contract_version=declared + 1)
            self.assertEqual(next_version.as_dict()["replay_contract_version"], declared + 1)
            self.assertNotEqual(next_version.as_dict(), new_arm.as_dict())

    def test_strategy_consumer_rejects_superseded_primary_dl_source(self):
        source = get_strategy_comparison_settings(
            STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        ).dl_sources["CONT13CP_ROLL"]
        workflow = SimpleNamespace(point_in_time_fold_months=12)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _write_rolling(path, VERSIONED_PROFILE, legacy=True)
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
                _write_rolling(path, VERSIONED_PROFILE)
                self.assertIs(load_validated_selection_pit_strategy_compare_contract(
                    root=path, source=source, workflow=workflow, seed=42,
                    point_in_time_dir_override=path,
                ), record)

    def test_versioned_positive_auxiliary_invalidates_only_dependent_pairs(self):
        from core.breakout_quality_fit_contract_version import current_fit_contract_artifact_identity
        from services.research.strategy_compare_reuse import (
            _versioned_replay_sources_pending, _pair_fit_lineages_match,
            _pair_cache_fingerprint_from_payload,
        )
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        lineage = current_fit_contract_artifact_identity(VERSIONED_PROFILE)
        self.assertEqual(lineage["version"], 2)
        self.assertIsNone(current_fit_contract_artifact_identity(UNAFFECTED_PROFILE))
        old = {}
        current = {"dl:CONT13CP_ROLL:fit_contract": lineage}
        state = {"artifact_identities": current, "dl_sources": {
            "CONT13CO_ROLL": {"ready": True},
            "CONT13CP_ROLL": {"ready": False},
        }}
        for arm_id in ("C125", "C126"):
            arm = settings.arms[arm_id]
            self.assertEqual(
                _versioned_replay_sources_pending(settings=settings, status=state, on_arm=arm),
                ("CONT13CP_ROLL",),
            )
            self.assertFalse(_pair_fit_lineages_match(
                settings=settings, on_arm=arm, current=current, published=old,
            ))
            kwargs = dict(
                settings_payload=settings.as_dict(),
                comparison_period={"start": "2021-01-01", "end": "2026-03-02"},
                off_arm_payload=settings.arms["C96"].as_dict(),
                on_arm_payload=arm.as_dict(), engine_schema_version=12,
                parameter_evaluation_sha256="a" * 64,
            )
            previous = _pair_cache_fingerprint_from_payload(artifact_identities=old, **kwargs)
            upgraded = _pair_cache_fingerprint_from_payload(artifact_identities=current, **kwargs)
            self.assertNotEqual(previous, upgraded)  # Even identical score columns.
            state["dl_sources"]["CONT13CP_ROLL"]["ready"] = True
            self.assertEqual(_versioned_replay_sources_pending(
                settings=settings, status=state, on_arm=arm), ())
            self.assertTrue(_pair_fit_lineages_match(
                settings=settings, on_arm=arm, current=current, published=dict(current),
            ))
            state["dl_sources"]["CONT13CP_ROLL"]["ready"] = False
        for arm_id in ("C123", "C124"):
            arm = settings.arms[arm_id]
            self.assertEqual(_versioned_replay_sources_pending(
                settings=settings, status=state, on_arm=arm), ())
            self.assertTrue(_pair_fit_lineages_match(
                settings=settings, on_arm=arm, current=current, published=old,
            ))

    def test_comparison_planner_waits_for_versioned_auxiliary_then_reuses(self):
        from core.breakout_quality_fit_contract_version import current_fit_contract_artifact_identity
        from core.research_orchestration import ResearchArtifactPlan
        from services.research.strategy_compare_reuse import _collect_replay_cache_status
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        status = {
            "comparison_period": {"start": "2021-01-01", "end": "2026-03-02"},
            "preparation_plan": ResearchArtifactPlan.from_actions([]),
            "artifact_identities": {
                "dl:CONT13CP_ROLL:fit_contract": current_fit_contract_artifact_identity(VERSIONED_PROFILE),
            },
            "dl_sources": {
                "CONT13CO_ROLL": {"ready": True},
                "CONT13CP_ROLL": {"ready": False},
            },
        }
        # Simulate a cached v1 C125/C126 pair whose contents otherwise match;
        # the planner must not allow evidence to override a pending FIT version.
        def fake_cached_pair(*, on_arm, **_kwargs):
            return {"source_pair_dir": Path("synthetic_completed_pair")}
        with patch("services.research.strategy_compare_reuse._completed_reuse_run_dirs",
                   return_value=(Path("synthetic_completed_run"),)), \
             patch("services.research.strategy_compare_reuse._find_reusable_pair",
                   side_effect=fake_cached_pair), \
             patch("services.research.strategy_compare_reuse._find_reusable_baseline_source",
                   return_value=Path("synthetic_completed_baseline")):
            before = _collect_replay_cache_status(
                root=Path("synthetic"), settings=settings, status=status,
            )["arm_states"]
            for arm in ("C125", "C126"):
                self.assertEqual(before[arm]["action"], "WAIT")
            for arm in ("C123", "C124"):
                self.assertEqual(before[arm]["action"], "REUSE")
            status["dl_sources"]["CONT13CP_ROLL"]["ready"] = True
            after = _collect_replay_cache_status(
                root=Path("synthetic"), settings=settings, status=status,
            )["arm_states"]
            for arm in ("C125", "C126"):
                self.assertEqual(after[arm]["action"], "REUSE")

    def test_robustness_report_checks_own_seed_fit_not_main_rolling(self):
        from services.research.target_dl_overview import _robustness_fit_versions_current
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for index in range(2):
                directory = root / "robustness" / f"seed_{index}"
                _write_rolling(directory, VERSIONED_PROFILE, legacy=(index == 1))
                report = directory / "selection_point_in_time_audit.json"
                report.write_text("{}", encoding="utf-8")
                paths.append(report.relative_to(root).as_posix())
            source = {"source_reports": paths}
            self.assertFalse(_robustness_fit_versions_current(
                root, model=source, profile=VERSIONED_PROFILE, seeds=(101, 102)))
            _write_rolling((root / paths[1]).parent, VERSIONED_PROFILE)
            self.assertTrue(_robustness_fit_versions_current(
                root, model=source, profile=VERSIONED_PROFILE, seeds=(101, 102)))
            self.assertFalse(_robustness_fit_versions_current(
                root, model={"source_reports": paths[:1]}, profile=VERSIONED_PROFILE,
                seeds=(101, 102)))

    def test_robustness_rho_independent_of_main_rolling_version_state(self):
        """Valid per-Seed Robustness evidence must survive a missing main Rolling FIT."""
        from services.research import target_dl_overview as overview
        from core.audit_policy import get_reusable_audit_definitions
        source = next(d.source for d in get_reusable_audit_definitions("breakout_quality")
                      if d.audit_type == "target_ground_truth_strategy_romd")

        target = "DailyUniversal_MinTrade_NetPnL_Positive"
        rank_profile = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE).dl_sources[
            "CONT13CO_ROLL"].experiment_profile
        evidence = ((target,), ({"target": target, "positive_output": True, "selector": "exact"},),
                    {target: "daily_formal_mintrade_fullcycle_netpnl_per_seed_v1"},
                    {target: {"oracle_romd": 90.84, "win_rate_pct": 99.88}}, {})
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(overview, "load_report_contract_versions", return_value={}), \
             patch.object(overview, "get_formal_oracle_definition",
                          return_value=SimpleNamespace(source=source)), \
             patch.object(overview, "get_active_audit_module_id", return_value="synthetic"), \
             patch.object(overview, "validate_oracle_compare_family_coverage", return_value=None), \
             patch.object(overview, "_oracle_rows", return_value=evidence), \
             patch.object(overview, "_strategy_bindings", return_value={(target, rank_profile): "C125"}), \
             patch.object(overview, "_is_exact_supervision", side_effect=lambda profile, _label: profile == rank_profile), \
             patch.object(overview, "_model_comparison_rhos",
                          side_effect=[{}, {rank_profile: (0.1234, (101, 102, 103, 104))}]), \
             patch.object(overview, "_rolling_fit_version_current", return_value=False), \
             patch.object(overview, "_single_model_rho", return_value=None), \
             patch.object(overview, "_published_model_report", return_value=None), \
             patch.object(overview, "_strategy_rolling_metrics", return_value={}), \
             patch.object(overview, "_strategy_robustness_metrics", return_value={}):
            rows = overview.collect_target_dl_overview(Path(tmp))["rows"]
        # Per-seed ranking evidence remains valid without a completed main
        # Rolling FIT, but rho is owned by the ranker, never by CP classifier.
        rank_rows = [row for row in rows if row["model"] == "MR-13CO"]
        self.assertEqual(len(rank_rows), 1)
        self.assertEqual(rank_rows[0]["positive_model"], "MR-13CP")
        self.assertIsNone(rank_rows[0]["rolling_rho"])
        self.assertEqual(rank_rows[0]["robustness_rho"], 0.1234)
        self.assertEqual(rank_rows[0]["rho_seeds"], (101, 102, 103, 104))
        self.assertFalse(any(row["model"] == "MR-13CP" for row in rows))

    def test_rolling_standard_sop_is_read_from_output_not_model_fit_tree(self):
        """The approved negative Rolling rho survives a stale comparison inventory."""
        import hashlib
        from core.breakout_quality_registry import (
            get_continuous_ranker_research_spec,
        )
        from core.report_contract_versions import load_report_contract_versions
        from core.research_report_contract import persistent_report_contract_fingerprint
        from filters.breakout_quality.paths import (
            resolve_filter_point_in_time_dir, resolve_filter_model_output_dir,
        )
        from services.research.target_dl_overview import (
            _model_paths, _model_fit_publication_dir, _single_model_rho,
            _rolling_fit_version_current,
        )
        versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
        profile = UNAFFECTED_PROFILE
        filter_id = "breakout_quality_v1"
        architecture = get_breakout_quality_experiment_profile(profile).model_architecture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, markdown = _model_paths(root, filter_id=filter_id, profile=profile)
            self.assertEqual(
                report.parent,
                resolve_filter_model_output_dir(root, filter_id, architecture, profile) / "point_in_time",
            )
            self.assertNotEqual(
                report.parent,
                resolve_filter_point_in_time_dir(root, filter_id, architecture, profile),
            )
            self.assertEqual(
                _model_fit_publication_dir(root, filter_id=filter_id, profile=profile,
                                           evaluation_mode="rolling_oos"),
                resolve_filter_point_in_time_dir(root, filter_id, architecture, profile),
            )
            markdown.parent.mkdir(parents=True)
            markdown.write_text("approved negative Rolling metric\n", encoding="utf-8")
            _write_json(report, {
                "report_id": "model.standard_sop",
                "report_version": versions["model.standard_sop"][-1]["version"],
                "report_contract_fingerprint": persistent_report_contract_fingerprint("model.standard_sop"),
                "markdown_sha256": hashlib.sha256(markdown.read_text().encode()).hexdigest(),
                "profile": profile, "filter_id": filter_id,
                "model_research_id": get_continuous_ranker_research_spec(profile).model_research_id,
                "evaluation_mode": "rolling_oos",
                "standard_model_sop": {"evaluation_mode": "rolling_oos", "split_metrics": {
                    "oos": {"mean_daily_spearman": -0.05575566460937493},
                }},
            })
            self.assertTrue(_rolling_fit_version_current(root, filter_id=filter_id, profile=profile))
            self.assertAlmostEqual(_single_model_rho(
                report, markdown, filter_id=filter_id, profile=profile, versions=versions,
                fit_publication_dir=_model_fit_publication_dir(
                    root, filter_id=filter_id, profile=profile, evaluation_mode="rolling_oos",
                ),
            ), -0.05575566460937493)

    def test_old_model_and_strategy_results_blank_until_matching_v2(self):
        import hashlib
        from core.breakout_quality_fit_contract_version import current_fit_contract_artifact_identity
        from core.report_contract_versions import load_report_contract_versions
        from core.research_report_contract import persistent_report_contract_fingerprint
        from core.breakout_quality_registry import get_continuous_ranker_research_spec
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.research.target_dl_overview import (
            _published_model_report, _model_comparison_rhos,
            _arm_fit_lineage_current, _robust_arm_fit_lineage_current,
        )
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            publication = root / "model_rolling"
            _write_rolling(publication, VERSIONED_PROFILE, legacy=True)
            report_dir = root / "output_rolling"
            report_dir.mkdir()
            md = report_dir / "standard_model_report.md"
            md.write_text("valid synthetic Standard SOP\n", encoding="utf-8")
            payload = {
                "report_id": "model.standard_sop", "report_version": versions["model.standard_sop"][-1]["version"],
                "report_contract_fingerprint": persistent_report_contract_fingerprint("model.standard_sop"),
                "profile": VERSIONED_PROFILE, "filter_id": "breakout_quality_v1", "evaluation_mode": "rolling_oos",
                "model_research_id": get_continuous_ranker_research_spec(VERSIONED_PROFILE).model_research_id,
                "markdown_sha256": hashlib.sha256(md.read_text().encode()).hexdigest(),
                "standard_model_sop": {"evaluation_mode": "rolling_oos", "split_metrics": {"oos": {"mean_daily_spearman": 0.11}}},
            }
            report = report_dir / "standard_model_report.json"
            def read():
                return _published_model_report(
                    report, md, profile=VERSIONED_PROFILE, filter_id="breakout_quality_v1",
                    evaluation_mode="rolling_oos", versions=versions,
                    fit_publication_dir=publication,
                )
            _write_json(report, payload)
            self.assertIsNone(read())  # Old v1 report must not appear as v2.
            payload["fit_contract_version"] = 2
            _write_json(report, payload)
            self.assertIsNone(read())  # v1 Fold still invalid despite report metadata.
            _write_rolling(publication, VERSIONED_PROFILE)
            self.assertIsNotNone(read())

            source = settings.dl_sources["CONT13CP_ROLL"]
            lineage = current_fit_contract_artifact_identity(VERSIONED_PROFILE)
            with patch('services.research.target_dl_overview.resolve_strategy_compare_selection_pit_bundle_dir', return_value=publication):
                self.assertFalse(_arm_fit_lineage_current(
                    root=root, settings=settings, arm_id="C125", artifact_identities={},
                ))
                for arm in ("C125", "C126"):
                    self.assertTrue(_arm_fit_lineage_current(
                        root=root, settings=settings, arm_id=arm,
                        artifact_identities={"dl:CONT13CP_ROLL:fit_contract": lineage},
                    ))
                self.assertTrue(_arm_fit_lineage_current(
                    root=root, settings=settings, arm_id="C123", artifact_identities={},
                ))
            pinned = {"runtime_dl_sources": [
                {"dl_id": "CONT13CP_ROLL", "experiment_profile": VERSIONED_PROFILE,
                 "experiment_profile_manifest": _manifest_settings(VERSIONED_PROFILE, legacy=True)}
            ]}
            self.assertFalse(_robust_arm_fit_lineage_current(settings,"C125",pinned))
            pinned["runtime_dl_sources"][0]["experiment_profile_manifest"] = _manifest_settings(VERSIONED_PROFILE)
            # An otherwise-current FIT without the v2 Replay contract is not
            # a current C125 Robustness result.
            self.assertFalse(_robust_arm_fit_lineage_current(settings,"C125",pinned))
            pinned["replay_contract_version"] = settings.arms["C125"].replay_contract_version
            self.assertTrue(_robust_arm_fit_lineage_current(settings,"C125",pinned))
            self.assertTrue(_robust_arm_fit_lineage_current(settings,"C123",{}))

            out = resolve_filter_output_dir(root, filter_id="breakout_quality_v1") / "standard_model_comparison" / "rolling_oos"
            out.mkdir(parents=True)
            document = out / "standard_model_comparison.md"
            document.write_text("synthetic comparison", encoding="utf-8")
            compare = {
                "report_id": "model.standard_comparison",
                "report_version": versions["model.standard_comparison"][-1]["version"],
                "report_contract_fingerprint": persistent_report_contract_fingerprint("model.standard_comparison"),
                "evaluation_mode": "rolling_oos", "robustness": False,
                "models": [{
                    "model_id": get_continuous_ranker_research_spec(VERSIONED_PROFILE).model_research_id,
                    "filter_id": "breakout_quality_v1", "profile": VERSIONED_PROFILE, "seed": 42,
                    "sop_view": {"evaluation_mode": "rolling_oos", "learnability": [{
                        "split": "oos", "mean_daily_spearman": 0.15,
                    }]},
                }],
            }
            file = out / "standard_model_comparison.json"
            _write_json(file, compare)
            # The comparison reader also requires the actual registered Rolling FIT path.
            with patch('services.research.target_dl_overview._rolling_fit_version_current', return_value=True):
                self.assertEqual(_model_comparison_rhos(root,filter_id="breakout_quality_v1",
                    versions=versions,robustness=False),{})
                compare["models"][0]["fit_contract_version"] = 2
                _write_json(file, compare)
                self.assertEqual(_model_comparison_rhos(root,filter_id="breakout_quality_v1",
                    versions=versions,robustness=False)[VERSIONED_PROFILE],(0.15,(42,)))

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
        ).dl_sources["CONT13CP_ROLL"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "published"
            _write_rolling(target, VERSIONED_PROFILE, legacy=True)
            (target / "old_checkpoint").write_text("legacy model")
            captured = {}
            def fake_builder(**kwargs):
                captured.update(kwargs)
                return ["python", "fake"], [], {"score_source": kwargs["source"].score_source,
                    "score_start_date": "2021-01-01", "score_end_date": "2026-03-02"}
            def fake_train(**kwargs):
                self.assertFalse(target.exists())
                _write_rolling(target, VERSIONED_PROFILE)
                return {"elapsed_sec": 0.0}
            def fake_validator(**kwargs):
                # Only a committed current publication may be used by Strategy Compare.
                self.assertFalse((root / ".published.superseded_fit_backup").exists())
                self.assertEqual(published_fit_contract_version_state(
                    experiment_profile=VERSIONED_PROFILE, publication_dir=target, rolling=True,
                )[0], "CURRENT")
                return {"ready": True}
            with patch.object(strategy_compare_training, 'build_strategy_compare_trainer_command', side_effect=fake_builder), patch.object(strategy_compare_training, 'run_logged_training_process', side_effect=fake_train), patch.object(strategy_compare_training, 'validate_strategy_compare_training_artifacts', side_effect=fake_validator), patch('services.breakout_quality.point_in_time_audit.audit_selection_point_in_time_scores', return_value=0):
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
                experiment_profile=VERSIONED_PROFILE, publication_dir=target, rolling=True)[0], "CURRENT")


    def test_rolling_version_rebuild_does_not_rollback_completed_fit_for_audit_failure(self):
        """Regression: VERSIONED_PROFILE v2 six-fold FIT must survive post-training SOP/Audit work."""
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings, get_breakout_quality_rolling_test_mode

        settings = get_breakout_quality_workflow_settings(experiment_profile=VERSIONED_PROFILE)
        mode = get_breakout_quality_rolling_test_mode("rolling")
        for scenario in ("success", "audit_failure", "producer_failure", "report_failure"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "rolling"
                backup = path.with_name("." + path.name + ".superseded_fit_backup")
                _write_rolling(path, VERSIONED_PROFILE, legacy=True)
                calls = []

                def fake_command(command, args, **kwargs):
                    calls.append(command)
                    if command == "build-point-in-time-scores":
                        self.assertTrue(backup.is_dir(), "v1 backup needed during FIT")
                        self.assertFalse(path.exists(), "v1 cannot be reused by v2 FIT")
                        _write_rolling(path, VERSIONED_PROFILE)
                        return 7 if scenario == "producer_failure" else 0
                    self.assertEqual(command, "audit-point-in-time-scores")
                    self.assertFalse(backup.exists(), "readers must run after commit")
                    self.assertEqual(published_fit_contract_version_state(
                        experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True,
                    )[0], "CURRENT")
                    return 7 if scenario == "audit_failure" else 0

                def fake_read_report(*args, **kwargs):
                    self.assertFalse(backup.exists())
                    self.assertEqual(published_fit_contract_version_state(
                        experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True,
                    )[0], "CURRENT")
                    if scenario == "report_failure":
                        return None, None, "SOP evidence incomplete"
                    return SimpleNamespace(audit_path=path / "audit.json"), {"standard_model_sop": {"valid": True}}, None

                with patch.object(application, '_prepare_continuous_research_inputs', return_value=0), \
                     patch.object(application, '_rolling_mode_point_in_time_dir', return_value=path), \
                     patch.object(application, '_rolling_publication_dir', return_value=path), \
                     patch.object(application, '_compact_console_scope', side_effect=nullcontext), \
                     patch('services.breakout_quality.continuous_ranker_pipeline.same_workflow_pit_bundle_handoff', side_effect=nullcontext), \
                     patch.object(application, '_run_command', side_effect=fake_command), \
                     patch.object(application, '_load_reusable_rolling_standard_report', side_effect=fake_read_report), \
                     patch.object(application, '_emit_standard_model_sop_report') as render:
                    if scenario == "producer_failure":
                        with self.assertRaisesRegex(RuntimeError, "Force Rebuild失敗"):
                            application._run_continuous_pit_profile(
                                "research", model_id="MR-13CP", profile_name=VERSIONED_PROFILE,
                                mode=mode, force_rebuild=True,
                            )
                    elif scenario == "report_failure":
                        with self.assertRaisesRegex(RuntimeError, "Standard SOP仍不可REUSE"):
                            application._run_continuous_pit_profile(
                                "research", model_id="MR-13CP", profile_name=VERSIONED_PROFILE,
                                mode=mode, force_rebuild=True,
                            )
                    else:
                        code = application._run_continuous_pit_profile(
                            "research", model_id="MR-13CP", profile_name=VERSIONED_PROFILE,
                            mode=mode, force_rebuild=True,
                        )
                        self.assertEqual(code, 7 if scenario == "audit_failure" else 0)
                self.assertFalse(backup.exists())
                self.assertEqual(calls, ["build-point-in-time-scores"] if scenario == "producer_failure"
                                 else ["build-point-in-time-scores", "audit-point-in-time-scores"])
                self.assertEqual(published_fit_contract_version_state(
                    experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True,
                )[0], "UPGRADE_REQUIRED" if scenario == "producer_failure" else "CURRENT")
                self.assertEqual(render.call_count, 1 if scenario == "success" else 0)

    def test_forward_version_rebuild_verifies_after_backup_is_removed(self):
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings

        settings = get_breakout_quality_workflow_settings(experiment_profile=VERSIONED_PROFILE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir, report_dir = root / "model", root / "report"
            _write_json(model_dir / "manifest.json", {
                "experiment_profile": VERSIONED_PROFILE, "experiment_settings": _manifest_settings(VERSIONED_PROFILE, legacy=True),
            })
            report_dir.mkdir()
            (report_dir / "old.txt").write_text("v1")
            checked = []

            def fake_command(command, args, **kwargs):
                self.assertEqual(command, "train-continuous-ranker")
                self.assertTrue((root / ".model.superseded_fit_backup").exists())
                _write_json(model_dir / "manifest.json", {
                    "experiment_profile": VERSIONED_PROFILE, "experiment_settings": _manifest_settings(VERSIONED_PROFILE),
                })
                report_dir.mkdir()
                return 0

            def fake_loader(_settings):
                checked.append(True)
                if len(checked) == 1:
                    return None, "v1 stale"
                self.assertFalse((root / ".model.superseded_fit_backup").exists())
                self.assertFalse((root / ".report.superseded_fit_backup").exists())
                self.assertEqual(published_fit_contract_version_state(
                    experiment_profile=VERSIONED_PROFILE, publication_dir=model_dir, rolling=False,
                )[0], "CURRENT")
                return SimpleNamespace(report={}), None

            with patch.object(application, '_forward_publication_paths', return_value=(model_dir, report_dir)), \
                 patch.object(application, '_load_reusable_continuous_forward_contract', side_effect=fake_loader), \
                 patch.object(application, '_collect_continuous_research_input_plan', return_value=SimpleNamespace(blocked=False)), \
                 patch.object(application, '_render_continuous_research_input_plan'), \
                 patch.object(application, '_prepare_continuous_research_inputs', return_value=0), \
                 patch.object(application, '_run_command', side_effect=fake_command), \
                 patch.object(application, '_clear_continuous_forward_reuse_caches'):
                code, contract, action = application._ensure_continuous_forward_model_report(
                    "research", model_id="MR-13CP", settings=settings,
                    prompt_for_build=False, force_rebuild=True,
                )
            self.assertEqual((code, action), (0, "FORCE_REBUILD"))
            self.assertIsNotNone(contract)
            self.assertEqual(len(checked), 2)
            self.assertFalse((root / ".model.superseded_fit_backup").exists())
            self.assertFalse((root / ".report.superseded_fit_backup").exists())

    def test_strategy_compare_audit_error_keeps_completed_versioned_pit_fit(self):
        source = get_strategy_comparison_settings(
            STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
        ).dl_sources["CONT13CP_ROLL"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "rolling"
            _write_rolling(target, VERSIONED_PROFILE, legacy=True)

            def fake_builder(**kwargs):
                return ["python", "mock"], [], {
                    "score_source": kwargs["source"].score_source,
                    "score_start_date": "2021-01-01", "score_end_date": "2026-03-02",
                }
            def fake_train(**kwargs):
                _write_rolling(target, VERSIONED_PROFILE)
                return {"elapsed_sec": 0.0}
            def fake_audit(**kwargs):
                self.assertFalse((root / ".rolling.superseded_fit_backup").exists())
                self.assertEqual(published_fit_contract_version_state(
                    experiment_profile=VERSIONED_PROFILE, publication_dir=target, rolling=True,
                )[0], "CURRENT")
                return 1
            with patch.object(strategy_compare_training, 'build_strategy_compare_trainer_command', side_effect=fake_builder), \
                 patch.object(strategy_compare_training, 'run_logged_training_process', side_effect=fake_train), \
                 patch('services.breakout_quality.point_in_time_audit.audit_selection_point_in_time_scores', side_effect=fake_audit):
                with self.assertRaisesRegex(RuntimeError, "PIT audit失敗"):
                    strategy_compare_training.run_strategy_compare_training_unit(
                        project_root=root, source=source,
                        workflow=SimpleNamespace(point_in_time_fold_months=12), seed=42,
                        model_dir=target, research_dir=None,
                        log_path=root / "log", comparison_start="2021-01-01",
                        comparison_end="2026-03-02", point_in_time_dir_override=target,
                    )
            self.assertFalse((root / ".rolling.superseded_fit_backup").exists())
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=target, rolling=True,
            )[0], "CURRENT")



    def test_rolling_robustness_rebuild_keeps_completed_fit_on_audit_failure(self):
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings

        settings = get_breakout_quality_workflow_settings(experiment_profile=VERSIONED_PROFILE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "seed_693545351" / "rolling"
            _write_rolling(path, VERSIONED_PROFILE, legacy=True)
            backup = path.with_name("." + path.name + ".superseded_fit_backup")
            def dirs(_settings, *, seed, rolling):
                self.assertEqual(int(seed), 693545351)
                return (path if rolling else root / "forward"), root / "report"
            def fake_command(command, args, **kwargs):
                if command == "build-point-in-time-scores":
                    self.assertTrue(backup.exists())
                    _write_rolling(path, VERSIONED_PROFILE)
                    return 0
                self.assertEqual(command, "audit-point-in-time-scores")
                self.assertFalse(backup.exists())
                self.assertEqual(published_fit_contract_version_state(
                    experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True,
                )[0], "CURRENT")
                return 9  # Audit cannot destroy a completed FIT publication.
            with patch.object(application, '_model_robustness_dirs', side_effect=dirs), \
                 patch.object(application, '_compact_console_scope', side_effect=nullcontext), \
                 patch('services.breakout_quality.continuous_ranker_pipeline.same_workflow_pit_bundle_handoff', side_effect=nullcontext), \
                 patch.object(application, '_run_command', side_effect=fake_command):
                code = application._build_rolling_robustness_seed(
                    "research", settings, seed=693545351, force_rebuild=True,
                    inputs_prepared=True,
                )
            self.assertEqual(code, 9)
            self.assertFalse(backup.exists())
            self.assertEqual(published_fit_contract_version_state(
                experiment_profile=VERSIONED_PROFILE, publication_dir=path, rolling=True,
            )[0], "CURRENT")



class SummaryRobustnessEvidenceCases(TestCase):
    """A partial cross-model run must not hide complete per-model evidence."""

    def setUp(self):
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.training_policy import get_robustness_benchmark_policy_snapshot
        from config.robustness_benchmark import MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID

        self.settings = get_breakout_quality_workflow_settings(experiment_profile=UNAFFECTED_PROFILE)
        self.seeds = tuple(get_robustness_benchmark_policy_snapshot(
            MODEL_STRATEGY_ROBUSTNESS_BENCHMARK_ID)["resolved_seeds"])
        self.profile = self.settings.experiment_profile
        self.scopes = ("validation", "oos", "breakout_candidate_oos")

    def _audit(self, seed):
        rho = (self.seeds.index(seed) + 1) / 10
        return {
            "filter_id": self.settings.filter_id,
            "model_architecture": self.settings.model_architecture,
            "experiment_profile": self.profile,
            "training": {"seed": seed},
            "standard_model_sop": {
                "schema": "standard_model_sop_v10", "evaluation_mode": "rolling_oos",
                "training": {"objective": "synthetic"},
                "evaluation_truth": {
                    "horizon_bars": 40, "mfe_source": "target_favorable_return",
                    "adverse_source": "target_adverse_return_to_peak",
                    "joint_quality_id": "same_day_min_mfe_safety_percentile_v1",
                    "cohort_identity": {key: "f" * 64 for key in self.scopes},
                    "price_truth_identity": {key: "a" * 64 for key in self.scopes},
                },
                "split_metrics": {key: {
                    "group_count": 8, "mean_daily_spearman": rho,
                    "global_spearman_vs_raw_target": 0.2, "pairwise_concordance": 0.6,
                    "top_score_decile_raw_target_mean": 2.5,
                    "bottom_score_decile_raw_target_mean": 0.5,
                } for key in self.scopes},
                "upside_downside_alignment_evaluation": {key: {
                    "target_to_full_mfe_daily_spearman": 0.5,
                    "target_to_low_adverse_daily_spearman": 0.3,
                    "score_to_full_mfe_daily_spearman": 0.4,
                    "score_to_low_adverse_daily_spearman": 0.1,
                    "joint_quality_evaluation_id": "same_day_min_mfe_safety_percentile_v1",
                    "top_10pct": {"n": 8, "mfe_pct_mean": 20.0,
                                  "adverse_pct_mean": 4.0, "joint_quality_pct": 70.0},
                } for key in self.scopes},
                "mode_extensions": {"rolling": {
                    "fold_drift": {"drift_flag": False, "flagged_folds": []},
                }},
            },
        }

    def _source_loader(self, root, *, defect=None):
        from filters.breakout_quality.paths import resolve_model_research_robustness_seed_dirs
        from services.breakout_quality.rolling_model_source import RollingModelSource

        def load(**kwargs):
            seed = kwargs["expected_seed"]
            self.assertIn(seed, self.seeds)
            self.assertEqual(Path(kwargs["project_root"]), root)
            directory, _unused = resolve_model_research_robustness_seed_dirs(
                root, self.settings.filter_id, self.settings.model_architecture,
                self.profile, seed=seed, rolling=True,
            )
            self.assertEqual(kwargs["point_in_time_dir_override"], directory)
            audit = self._audit(seed)
            complete = SimpleNamespace(seed=seed, manifest={}, audit=audit)
            if seed == self.seeds[-1]:
                if defect == "score_only":
                    return RollingModelSource(complete, None, "Audit missing", None)
                if defect == "missing":
                    return RollingModelSource(None, None, "Audit missing", "SCORE missing")
                if defect == "wrong_seed":
                    complete.seed = seed + 1
                if defect == "incomplete_sop":
                    audit["standard_model_sop"]["split_metrics"]["oos"].pop("mean_daily_spearman")
                if defect == "different_cohort":
                    audit["standard_model_sop"]["evaluation_truth"]["cohort_identity"]["oos"] = "b" * 64
            return RollingModelSource(complete, complete, None, None)
        return load

    def test_summary_recovers_completed_model_before_cross_model_publication(self):
        from core.breakout_quality_registry import get_continuous_ranker_research_spec
        from core.report_contract_versions import load_report_contract_versions
        from core.research_report_contract import persistent_report_contract_fingerprint
        from filters.breakout_quality.paths import resolve_filter_output_dir
        from services.research import target_dl_overview as overview

        versions = load_report_contract_versions(Path(__file__).resolve().parents[2])
        model_id = get_continuous_ranker_research_spec(self.profile).model_research_id
        for previous_comparison in (False, True):
            with self.subTest(previous_comparison=previous_comparison), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                if previous_comparison:
                    directory = (resolve_filter_output_dir(root, filter_id=self.settings.filter_id)
                                 / "standard_model_comparison" / "rolling_oos_robustness")
                    contract = application.require_approved_report_contract("model.standard_comparison")
                    _write_json(directory / "standard_model_comparison.json", {
                        "report_id": contract.report_id, "report_version": contract.version,
                        "report_contract_fingerprint": persistent_report_contract_fingerprint(contract.report_id),
                        "evaluation_mode": "rolling_oos", "robustness": True, "models": [],
                    })
                    (directory / "standard_model_comparison.md").write_text("Previous comparison\n")
                before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
                with patch.object(overview, "get_breakout_quality_model_test_settings", return_value=SimpleNamespace(
                    model_profiles=((model_id, self.profile),))), \
                     patch.object(application, "load_rolling_model_source", side_effect=self._source_loader(root)) as load, \
                     patch.object(application, "pit_fold_fitting_settings_issues", return_value=()), \
                     patch.object(application, "_run_command", side_effect=AssertionError("Summary must stay read-only")), \
                     patch.object(application, "_write_standard_model_comparison_report", side_effect=AssertionError("No publication")):
                    actual = overview._model_comparison_rhos(
                        root, filter_id=self.settings.filter_id, versions=versions, robustness=True,
                    )
                expected = sum((i + 1) / 10 for i in range(len(self.seeds))) / len(self.seeds)
                self.assertEqual(actual[self.profile], (expected, self.seeds))
                self.assertEqual(tuple(call.kwargs["expected_seed"] for call in load.call_args_list), self.seeds)
                self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()})

    def test_read_through_uses_identical_workflow_evidence_and_aggregation(self):
        from services.breakout_quality.model_artifact_readiness import resolve_model_robustness_readiness
        from services.breakout_quality.standard_model_sop import aggregate_standard_model_sop_robustness

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(application, "load_rolling_model_source", side_effect=self._source_loader(root)), \
                 patch.object(application, "pit_fold_fitting_settings_issues", return_value=()):
                sources = tuple(application._inspect_rolling_robustness_seed(
                    self.settings, seed=seed, project_root=root,
                ) for seed in self.seeds)
                readiness = resolve_model_robustness_readiness(mode="rolling", sources=sources)
                result, reason = application.load_completed_rolling_model_robustness(
                    self.settings, project_root=root, seeds=self.seeds,
                )
            self.assertEqual(readiness.display_action, "REUSE")
            self.assertIsNone(reason)
            expected = aggregate_standard_model_sop_robustness(
                [dict(source.verified_evidence["standard_model_sop"]) for source in sources],
                seeds=self.seeds,
            )
            self.assertEqual(result, expected)

    def test_incomplete_wrong_seed_or_incompatible_evidence_stays_blank(self):
        for defect in ("missing", "score_only", "wrong_seed", "incomplete_sop", "different_cohort", "stale_fit"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.object(application, "load_rolling_model_source", side_effect=self._source_loader(root, defect=defect)), \
                     patch.object(application, "pit_fold_fitting_settings_issues", return_value=("stale FIT",) if defect == "stale_fit" else ()), \
                     patch.object(application, "_run_command", side_effect=AssertionError("No rebuild")):
                    result, reason = application.load_completed_rolling_model_robustness(
                        self.settings, project_root=root, seeds=self.seeds,
                    )
                self.assertIsNone(result)
                self.assertTrue(reason)


if __name__ == "__main__":
    main()
