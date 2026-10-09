"""Focused in-memory/disk-fixture verification of the Promotion MVP.

Run independently: python -m unittest tools.validate.test_strategy_promotion_mvp
Never invokes the formal test suite or requires market Parquet data.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config.strategy_compare import STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE
from core.file_integrity import compute_file_sha256
from core.strategy_compare_policy import get_strategy_comparison_settings
from core.strategy_promotion import (
    approve_current_strategy_rule, current_strategy_rules, get_approved_rule,
    get_trading_release, load_promotion_approvals, load_trading_release_state,
    stage_trading_strategy_rule,
)
from core.trading_policy import (
    build_trading_strategy_param_training_plan, get_trading_strategy_profile,
    resolve_trading_selected_strategy_param_path,
)
from services.trading.strategy_promotion import (
    activate_staged_trading_release, assert_active_trading_release_ready,
)


class StrategyPromotionMVPTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        self.output_root = self.root / settings.output_root
        run = self.output_root / "runs" / "synthetic-complete"
        latest = self.output_root / "latest"
        result = {
            "status": "COMPLETED", "config_fingerprint": "synthetic-rolling-fingerprint",
            "settings": {"suite_id": settings.suite_id},
            "scenarios": {arm.arm_id: {"status": "COMPLETED"} for arm in settings.enabled_arms},
        }
        manifest = {
            "config_fingerprint": "synthetic-rolling-fingerprint",
            "enabled_arms": [arm.as_dict() for arm in settings.enabled_arms],
        }
        for directory in (run, latest):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "strategy_comparison.json").write_text(json.dumps(result), encoding="utf-8")
            (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_all_options_exactly_match_compare_suite(self):
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        self.assertEqual(
            [arm["arm_id"] for arm in current_strategy_rules()],
            [arm.arm_id for arm in settings.enabled_arms],
        )
        self.assertFalse(next(arm for arm in current_strategy_rules() if arm["arm_id"] == "C61")["trading_supported"])

    def test_approval_requires_rolling_evidence_and_pins_run(self):
        approval = approve_current_strategy_rule(self.root, "C95")
        self.assertIn("/runs/", approval["evidence_path"])
        self.assertIn("/runs/", approval["manifest_path"])
        self.assertEqual(get_approved_rule(self.root, "C95")["rule_sha256"], approval["rule_sha256"])
        (self.output_root / "latest" / "strategy_comparison.json").write_text("{}", encoding="utf-8")
        self.assertEqual(get_approved_rule(self.root, "C95")["rule_sha256"], approval["rule_sha256"])
        (self.root / approval["evidence_path"]).write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            get_approved_rule(self.root, "C95")

    def test_research_evidence_rejects_other_arm_manifest_and_tampering(self):
        manifest = self.output_root / "latest" / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["enabled_arms"] = payload["enabled_arms"][:-1]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "strategy arms differ"):
            approve_current_strategy_rule(self.root, "C95")
        # Restore the exact producer contract before approval.
        settings = get_strategy_comparison_settings(STRATEGY_COMPARE_CURRENT_ROLLING_PROFILE)
        payload["enabled_arms"] = [arm.as_dict() for arm in settings.enabled_arms]
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        approval = approve_current_strategy_rule(self.root, "C95")
        pinned_manifest = self.root / approval["manifest_path"]
        pinned_manifest.write_text("{}", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            get_approved_rule(self.root, "C95")

    def test_frozen_pending_market_clock_does_not_require_new_scanner(self):
        from services.trading.pending_entry_service import _latest_finalized_date_for_frozen_pending
        with patch("services.trading.pending_entry_service.load_trading_v2_consumer_state", return_value={"market_date": "2026-10-09"}), \
             patch("services.trading.pending_entry_service.load_trading_scanner_runtime", side_effect=RuntimeError("new release blocked")):
            self.assertEqual(_latest_finalized_date_for_frozen_pending(self.root), "2026-10-09")

    def test_unapproved_and_unsupported_are_not_staged(self):
        with self.assertRaises(RuntimeError):
            stage_trading_strategy_rule(self.root, "C95")
        approve_current_strategy_rule(self.root, "C61")
        with self.assertRaisesRegex(RuntimeError, "multi-seed"):
            stage_trading_strategy_rule(self.root, "C61")
        self.assertIsNone(load_trading_release_state(self.root)["pending"])

    def test_staged_fitting_is_isolated_until_explicit_activation(self):
        approve_current_strategy_rule(self.root, "C95")
        old_profile = get_trading_strategy_profile(self.root)
        old_path = resolve_trading_selected_strategy_param_path(self.root)
        staged = stage_trading_strategy_rule(self.root, "C95")
        self.assertEqual(get_trading_strategy_profile(self.root).strategy_param_profile_id, old_profile.strategy_param_profile_id)
        self.assertEqual(resolve_trading_selected_strategy_param_path(self.root), old_path)
        plan = build_trading_strategy_param_training_plan(self.root)
        from core.strategy_promotion import current_strategy_rule
        selected_arm = current_strategy_rule("C95")
        self.assertEqual(plan["strategy_param_profile_id"], selected_arm["training_profile_id"])
        self.assertEqual(plan["parameter_set_identity"], selected_arm["parameter_set_identity"])
        self.assertEqual(plan["optimizer_seed_mode"], "random_n_each_training_run")
        self.assertEqual(plan["optimizer_seed_count"], 4)
        self.assertEqual(plan["optimizer_trials_per_seed"], 300)
        self.assertEqual(plan["release_id"], staged["release_id"])
        self.assertIn(staged["release_id"], plan["selected_params_path"])
        self.assertNotEqual(str(plan["selected_params_path"]), str(old_path))
        self.assertFalse(load_trading_release_state(self.root)["active"])
        selected = Path(plan["selected_params_path"])
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text("synthetic candidate params", encoding="utf-8")
        from services.trading.strategy_promotion import _param_manifest_path
        manifest = _param_manifest_path(self.root, purpose="training")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("synthetic native manifest", encoding="utf-8")
        binding = {"promotion_release_id": staged["release_id"], "strategy_rule_sha256": staged["rule_sha256"]}
        with patch("services.trading.strategy_promotion.load_trading_strategy_param_binding", return_value=binding), \
             patch("services.trading.strategy_promotion.get_trading_strategy_param_binding_sha256", return_value="synthetic-binding-sha"):
            activated = activate_staged_trading_release(self.root)
            self.assertEqual(activated["arm_id"], "C95")
            self.assertEqual(get_trading_strategy_profile(self.root).parameter_set_identity, "PARAM-P14")
            self.assertEqual(resolve_trading_selected_strategy_param_path(self.root), str(selected))
            self.assertIsNone(load_trading_release_state(self.root)["pending"])
            self.assertEqual(assert_active_trading_release_ready(self.root)["release_id"], staged["release_id"])
            selected.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Params 已改變"):
                assert_active_trading_release_ready(self.root)

    def test_canonical_optimizer_receives_stage_namespace_without_fixed_seed(self):
        """No production optimizer execution; verifies the real producer call boundary."""
        from services.trading.strategy_param_training import run_trading_strategy_param_training
        approve_current_strategy_rule(self.root, "C95")
        pending = stage_trading_strategy_rule(self.root, "C95")
        plan = build_trading_strategy_param_training_plan(self.root)
        observed = {}

        def producer(**kwargs):
            observed.update(kwargs)
            self.assertEqual(Path(kwargs["strategy_params_root"]), Path(plan["strategy_params_root"]))
            self.assertEqual(kwargs["seed_ensemble_policy_override"]["seed_mode"], "random_n_each_training_run")
            self.assertNotIn("resolved_seeds", kwargs["seed_ensemble_policy_override"])
            selected = Path(plan["selected_params_path"])
            selected.parent.mkdir(parents=True, exist_ok=True)
            selected.write_text("independently selected", encoding="utf-8")
            manifest = Path(plan["output_dir"]) / "manifest.json"
            return {"selected_params_path": str(selected), "manifest_path": str(manifest), "resolved_optimizer_seeds": [3, 5, 7, 11]}

        def binding_publisher(root, *, usage_mode, purpose):
            self.assertEqual(purpose, "training")
            self.assertEqual(str(root), str(self.root.resolve()))
            return {
                "selected_params_sha256": compute_file_sha256(plan["selected_params_path"]),
                "binding_fingerprint": "synthetic-binding",
                "usage_mode": usage_mode,
                "param_training_data_date": "2026-10-09",
                "param_member_count": 6, "param_min_agree": 3,
            }

        consumer = {"market_date": "2026-10-09", "source_view_fingerprint": "synthetic-source"}
        with patch("services.trading.strategy_param_training.load_trading_v2_consumer_state", return_value=consumer), \
             patch("services.trading.strategy_param_training.get_trading_v2_consumer_state_sha256", return_value="synthetic-state"), \
             patch("services.trading.strategy_param_training.run_static_strategy_parameter_training", side_effect=producer), \
             patch("services.trading.strategy_param_training.publish_trading_strategy_param_binding", side_effect=binding_publisher):
            result = run_trading_strategy_param_training(project_root=self.root)
        from core.strategy_promotion import current_strategy_rule
        self.assertEqual(result["strategy_param_profile_id"], current_strategy_rule("C95")["training_profile_id"])
        self.assertEqual(result["optimizer_resolved_seeds"], [3, 5, 7, 11])
        self.assertIn(pending["release_id"], observed["output_dir"])
        self.assertIsNone(load_trading_release_state(self.root)["active"])
        self.assertEqual(get_trading_strategy_profile(self.root).parameter_set_identity, "PARAM-P6")

    def test_no_failed_activation_can_overwrite_active_release(self):
        approve_current_strategy_rule(self.root, "C95")
        stage = stage_trading_strategy_rule(self.root, "C95")
        with self.assertRaises(RuntimeError):
            activate_staged_trading_release(self.root)
        state = load_trading_release_state(self.root)
        self.assertIsNone(state["active"])
        self.assertEqual(state["pending"]["release_id"], stage["release_id"])

    def test_legacy_trading_is_unchanged_until_promotion(self):
        self.assertIsNone(assert_active_trading_release_ready(self.root))
        self.assertIsNone(load_trading_release_state(self.root)["active"])
        self.assertEqual(get_trading_strategy_profile(self.root).parameter_set_identity, "PARAM-P6")


if __name__ == "__main__":
    unittest.main()
