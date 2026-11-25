"""Execution menus expose each canonical seed decision without long identities."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.console_report import render_table
from core.display_common import _display_width, _strip_ansi
from core.research_status_display import render_seed_status_matrix, render_dependency_summary
from services.research import breakout_quality_application as model


class ResearchSeedStatusDisplayTests(unittest.TestCase):
    def test_matrix_preserves_config_order_and_mixed_seed_decisions(self):
        rendered = _strip_ansi(render_seed_status_matrix(
            "策略", seeds=(91, 7, 42),
            rows=(("C123", ("REUSE", "RUN", "WAIT")),
                  ("C106", ("RUN", "REUSE", "REUSE"))),
        ))
        self.assertIn("S1=91  S2=7  S3=42", rendered)
        rows = [line.split() for line in rendered.splitlines() if line.startswith("C")]
        self.assertEqual(rows, [["C123", "REUSE", "RUN", "WAIT"],
                                ["C106", "RUN", "REUSE", "REUSE"]])

    def test_missing_or_duplicate_seed_information_cannot_look_complete(self):
        for seeds, rows in (((91, 7), (("C123", ("REUSE",)),)),
                            ((91, 91), (("C123", ("REUSE", "RUN")),)),
                            ((91,), (("C123", ("",)),)),
                            ((91,), (("C123", ("RUN",)), ("C123", ("REUSE",))))):
            with self.subTest(seeds=seeds, rows=rows), self.assertRaises(ValueError):
                render_seed_status_matrix("策略", seeds=seeds, rows=rows)

    def test_model_uses_per_seed_states_and_keeps_report_refresh_note(self):
        def seed_state(seed, action, reason):
            return SimpleNamespace(seed=seed, readiness=SimpleNamespace(
                display_action=action, display_reason=reason))
        readiness = SimpleNamespace(seeds=(
            seed_state(42, "BUILD/REFRESH", "audit missing"),
            seed_state(91, "REUSE", "READY"),
            seed_state(7, "REUSE", "READY; report artifact refresh only"),
        ))
        with redirect_stdout(io.StringIO()) as output:
            model._print_model_action_status(
                [("MR-13CO", "long_internal_profile", "REUSE", "aggregate")],
                seeds=(91, 7, 42), seed_readiness=(readiness,),
            )
        rendered = _strip_ansi(output.getvalue())
        self.assertIn("S1=91  S2=7  S3=42", rendered)
        first_model_row = next(line.split() for line in rendered.splitlines()
                               if line.startswith("MR-13CO"))
        self.assertEqual(first_model_row, ["MR-13CO", "REUSE", "REUSE", "BUILD/REFRESH"])
        self.assertIn("MR-13CO S2", rendered)
        self.assertIn("report artifact refresh only", rendered)
        self.assertIn("MR-13CO S3", rendered)
        self.assertNotIn("long_internal_profile", rendered)
        self.assertNotIn("aggregate", rendered)

    def test_model_rejects_incomplete_seed_inventory(self):
        readiness = SimpleNamespace(seeds=())
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            model._print_model_action_status(
                [("MR-13CO", "profile", "REUSE", "READY")],
                seeds=(91, 7), seed_readiness=(readiness,),
            )

    def test_common_refresh_note_does_not_repeat_every_model_and_seed(self):
        ready = SimpleNamespace(seeds=tuple(
            SimpleNamespace(seed=seed, readiness=SimpleNamespace(
                display_action="REUSE", display_reason="READY; report artifact refresh only"))
            for seed in (91, 7)
        ))
        with redirect_stdout(io.StringIO()) as output:
            model._print_model_action_status(
                [("MR-A", "profile_a", "REUSE", "READY"),
                 ("MR-B", "profile_b", "REUSE", "READY")],
                seeds=(91, 7), seed_readiness=(ready, ready),
            )
        rendered = _strip_ansi(output.getvalue())
        self.assertIn("全部模型／Seed", rendered)
        self.assertEqual(rendered.count("report artifact refresh only"), 1)
        self.assertNotIn("MR-A S1", rendered)

    def test_reused_dependencies_are_counted_and_pending_reasons_are_complete(self):
        reason = "canonical parameter missing; " + "完整原因" * 35
        with patch("core.research_status_display.render_table",
                   wraps=render_table) as table:
            rendered = _strip_ansi(render_dependency_summary([
                ("REUSE", "very_long/internal/path", "ready verbose identity"),
                ("BUILD", "P16 S1", reason),
                ("BUILD", "P16 S2", reason),
            ]))
        self.assertIn("REUSE 1 / BUILD 2", rendered)
        self.assertNotIn("very_long/internal/path", rendered)
        self.assertNotIn("ready verbose identity", rendered)
        details = table.call_args.args[1]
        self.assertEqual(details[0][0], "P16 S1 / P16 S2")
        self.assertEqual(details[0][2], reason)
        self.assertLessEqual(max(map(_display_width, rendered.splitlines())), 100)

    def test_shared_reason_refers_to_matching_cells_without_relisting_every_seed(self):
        def readiness(*actions):
            return SimpleNamespace(seeds=tuple(
                SimpleNamespace(seed=seed, readiness=SimpleNamespace(
                    display_action=action,
                    display_reason="verify checkpoint" if action == "VERIFY FIT" else "READY"))
                for seed, action in zip((91, 7), actions)
            ))
        with redirect_stdout(io.StringIO()) as output:
            model._print_model_action_status(
                [("MR-A", "profile_a", "VERIFY FIT", ""),
                 ("MR-B", "profile_b", "VERIFY FIT", "")],
                seeds=(91, 7), seed_readiness=(readiness("VERIFY FIT", "VERIFY FIT"),
                                             readiness("REUSE", "VERIFY FIT")),
            )
        rendered = _strip_ansi(output.getvalue())
        self.assertIn("表內 VERIFY FIT（3 格）", rendered)
        self.assertEqual(rendered.count("verify checkpoint"), 1)
        self.assertNotIn("MR-A S1", rendered)


if __name__ == "__main__":
    unittest.main()
