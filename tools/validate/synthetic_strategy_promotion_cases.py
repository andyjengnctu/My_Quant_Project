"""Strategy promotion and activation capability regression fixtures.

Run independently: python -m unittest tools.validate.synthetic_strategy_promotion_cases
Does not require market Parquet data or invoke the formal test suite.
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


class StrategyPromotionDialogAppearanceTest(unittest.TestCase):
    """Use a real Tk Treeview to guard against white-on-white native rows."""

    def test_dark_rows_and_live_research_approval_refresh(self):
        import tkinter as tk
        from tkinter import ttk

        from services.workbench_ui.trading_account_panel import TradingAccountPanel
        from services.workbench_ui.workbench import (
            configure_workbench_theme, WORKBENCH_SURFACE, WORKBENCH_TEXT,
            WORKBENCH_TREE_STYLE,
        )

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.addCleanup(root.destroy)
        configure_workbench_theme(root)
        panel = tk.Frame(root)
        panel.pack()
        panel._command_thread = None
        panel._param_mode_var = tk.StringVar(master=root, value="沿用既有 Params")
        panel._param_mode_user_selected = False
        panel._operations_snapshot = {}
        dashboard_updates = []
        panel._render_strategy_params_overview = (
            lambda snapshot, *, release_state: dashboard_updates.append(dict(release_state))
        )

        rules = (
            {"arm_id": "C58", "name": "Single Best", "trading_supported": False,
             "trading_blockers": ("single seed",), "rule_sha256": "1" * 64},
            {"arm_id": "C61", "name": "Full Best", "trading_supported": False,
             "trading_blockers": ("single seed",), "rule_sha256": "2" * 64},
            {"arm_id": "C95", "name": "Same-Signal Agree", "trading_supported": True,
             "trading_blockers": (), "rule_sha256": "3" * 64},
        )
        state = {"approved": False, "invalid": False}

        def approval(ignored_root, arm_id):
            if arm_id == "C95" and state["approved"]:
                return {"rule_sha256": "3" * 64}
            raise RuntimeError("Research approval evidence missing or invalid")

        def approval_records(ignored_root):
            return {"approvals": {"C95": {"status": "APPROVED"}} if state["invalid"] else {}}

        with patch("services.workbench_ui.trading_account_panel_promotion.current_strategy_rules", return_value=rules), \
             patch("services.workbench_ui.trading_account_panel_promotion.get_approved_rule", side_effect=approval), \
             patch("services.workbench_ui.trading_account_panel_promotion.load_promotion_approvals", side_effect=approval_records), \
             patch("services.workbench_ui.trading_account_panel_promotion.load_trading_release_state", return_value={"active": None, "pending": None}):
            TradingAccountPanel._open_strategy_promotion_dialog(panel)
            root.update()

            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)

            dialogs = [item for item in descendants(root) if isinstance(item, tk.Toplevel)]
            self.assertEqual(len(dialogs), 1)
            dialog = dialogs[0]
            widgets = list(descendants(dialog))
            tree = next(item for item in widgets if isinstance(item, ttk.Treeview))
            refresh_button = next(
                item for item in widgets if isinstance(item, ttk.Button) and item.cget("text") == "重新整理核准狀態"
            )
            stage_button = next(
                item for item in widgets if isinstance(item, ttk.Button) and item.cget("text") == "準備選取策略"
            )
            self.assertEqual(tree.cget("style"), WORKBENCH_TREE_STYLE)
            style = ttk.Style(root)
            self.assertEqual(style.lookup(tree.cget("style"), "background"), WORKBENCH_SURFACE)
            self.assertEqual(style.lookup(tree.cget("style"), "foreground"), WORKBENCH_TEXT)
            self.assertEqual(tree.get_children(), ("C58", "C61", "C95"))
            self.assertEqual(tree.selection(), ("C95",))
            self.assertEqual(tree.set("C95", "approval"), "待核准")
            self.assertEqual(str(stage_button.cget("state")), "disabled")
            self.assertEqual(dashboard_updates[-1], {"active": None, "pending": None})

            # Research [8] approval in another process, while the dialog stays open.
            state["approved"] = True
            refresh_button.invoke()
            self.assertEqual(tree.set("C95", "approval"), "已核准")
            self.assertEqual(tree.selection(), ("C95",))
            self.assertEqual(str(stage_button.cget("state")), "normal")

            # An old approval record whose scientific evidence drifted is invalid,
            # not an ordinary request that Research still needs to approve.
            state["approved"] = False
            state["invalid"] = True
            dialog.event_generate("<FocusIn>")
            root.update()
            self.assertEqual(tree.set("C95", "approval"), "核准失效")
            self.assertEqual(str(stage_button.cget("state")), "disabled")
            self.assertEqual(tree.get_children(), ("C58", "C61", "C95"))


class StrategyPromotionDashboardIdentityTest(unittest.TestCase):
    """The fixed Trading dashboard, not the daily action bar, owns Release status."""

    def test_active_and_pending_release_display_without_engine_id_duplication(self):
        from types import SimpleNamespace
        from services.workbench_ui.trading_account_panel import TradingAccountPanel

        updates = []
        panel = SimpleNamespace(_set_overview_card=lambda *a, **kw: updates.append((a, kw)))
        snapshot = {
            "strategy_id": "full_rule_based_no_dl",
            "param_selector": "base_finalists_agree",
            "param_usage_mode": "reuse_existing",
            "param_training_data_date": "2026-10-08",
            "params_ready_for_scan": True,
        }
        state = {
            "active": {"arm_id": "C95", "status": "ACTIVE"},
            "pending": {"arm_id": "C99", "status": "PENDING_PARAMS"},
        }
        with patch("services.workbench_ui.trading_account_panel_dashboard.get_trading_policy_snapshot", return_value={}), \
             patch("services.workbench_ui.trading_account_panel_dashboard.load_trading_release_state", return_value=state):
            TradingAccountPanel._render_strategy_params_overview(panel, snapshot)
        (key, primary, detail), styles = updates[-1]
        self.assertEqual(key, "strategy_params")
        self.assertEqual(primary, "ACTIVE C95")
        self.assertIn("C99", detail)
        self.assertIn("base_finalists_agree", detail)
        self.assertIn("2026-10-08", detail)
        self.assertNotIn("full_rule_based_no_dl", primary)
        self.assertEqual(styles["tone"], "success")

        with patch("services.workbench_ui.trading_account_panel_dashboard.get_trading_policy_snapshot", return_value={}):
            TradingAccountPanel._render_strategy_params_overview(
                panel, snapshot, release_state={"active": None, "pending": None}
            )
            self.assertNotIn("ACTIVE", updates[-1][0][1])
            TradingAccountPanel._render_strategy_params_overview(
                panel, snapshot, release_state={"active": {"arm_id": "C95", "status": "BROKEN"}, "pending": None}
            )
        self.assertEqual(updates[-1][1]["tone"], "error")

    def test_invalid_canonical_release_read_does_not_show_stale_active(self):
        from types import SimpleNamespace
        from services.workbench_ui.trading_account_panel import TradingAccountPanel

        updates = []
        panel = SimpleNamespace(_set_overview_card=lambda *a, **kw: updates.append((a, kw)))
        with patch("services.workbench_ui.trading_account_panel_dashboard.get_trading_policy_snapshot", return_value={}), \
             patch("services.workbench_ui.trading_account_panel_dashboard.load_trading_release_state", side_effect=ValueError("invalid fingerprint")):
            TradingAccountPanel._render_strategy_params_overview(panel, {})
        self.assertIn("Release", updates[-1][0][1])
        self.assertEqual(updates[-1][1]["tone"], "error")


def validate_trading_strategy_promotion_capability_case(_base_params):
    """In-memory promotion / frozen Params contract and headless Trading dashboard."""
    from .check_result_utils import run_unittest_capability_cases

    return run_unittest_capability_cases(
        "TRADING_STRATEGY_PROMOTION_CAPABILITY", "trading_strategy_promotion",
        (StrategyPromotionMVPTest, StrategyPromotionDashboardIdentityTest),
    )


if __name__ == "__main__":
    unittest.main()
