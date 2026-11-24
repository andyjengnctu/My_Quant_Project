"""Numerical source closure: report refactors stay warm, real arithmetic does not."""
from pathlib import Path
from unittest import TestCase
from core.file_integrity import canonical_json_sha256

from services.research.oracle_numerical_sources import oracle_build_source_parts_from_reader


class OracleNumericalSourceTests(TestCase):
    def setUp(self):
        # The formal capability runner executes individual cases; initialize
        # the immutable source fixture there as well as in unittest suites.
        if not hasattr(type(self), "original"):
            type(self).setUpClass()

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.original = {}
        cls.baseline = cls.parts({})

    @classmethod
    def parts(cls, replacements):
        def read(relative):
            if relative in replacements:
                value = replacements[relative]
                if value is None:
                    raise FileNotFoundError(relative)
                return value
            if relative not in cls.original:
                cls.original[relative] = (cls.root / relative).read_bytes()
            return cls.original[relative]
        return oracle_build_source_parts_from_reader(read)

    def test_unused_helper_and_report_layout_do_not_change_numerical_proof(self):
        owner = "filters/breakout_quality/strategy_compare_reporting.py"
        original = self.original[owner].decode("utf-8-sig")
        changed = original.replace(
            "def _markdown_report(metadata, baseline, quality, delta, yearly, strategy_diagnostics=None) -> str:",
            "def _markdown_report(metadata, baseline, quality, delta, yearly, strategy_diagnostics=None) -> str:\n    layout_only = 'new report layout'",
        )
        self.assertNotEqual(original, changed)
        changed += "\ndef unused_numerical_helper(value):\n    return value + 123456\n"
        self.assertEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(self.parts({owner: changed.encode("utf-8")})))

    def test_selected_arithmetic_and_native_source_invalidate(self):
        owner = "core/order_lot_policy.py"
        original = self.original[owner]
        changed = original.replace(b"return price_to_milli(limit_price) * qty_int",
                                   b"return price_to_milli(limit_price) + qty_int")
        self.assertNotEqual(original, changed)
        self.assertNotEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(self.parts({owner: changed})))
        owner = "core/native_optimizer_hotpath.c"
        original = (self.root / owner).read_bytes()
        changed = original.replace(b"1000", b"1001", 1)
        self.assertNotEqual(original, changed)
        self.assertNotEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(self.parts({owner: changed})))

    def test_new_imported_numerical_helper_is_followed_and_missing_owner_blocks(self):
        owner = "services/audit/oracle_observation.py"
        original = self.original[owner].decode("utf-8-sig")
        changed = original.replace('"romd": float(summary["return_over_max_drawdown"]),',
            '"romd": _reviewed_helper(summary["return_over_max_drawdown"]),')
        self.assertNotEqual(original, changed)
        changed += "\nfrom core.oracle_test_helper import numerical_value as _reviewed_helper\n"
        helper = "core/oracle_test_helper.py"
        sources = {owner: changed.encode(), helper: b"def numerical_value(value):\n    return float(value)\n"}
        projected = self.parts(sources)
        self.assertIn(helper, projected)
        sources[helper] = b"def numerical_value(value):\n    return float(value) + 1\n"
        self.assertNotEqual(canonical_json_sha256(projected), canonical_json_sha256(self.parts(sources)))
        sources[helper] = None
        with self.assertRaisesRegex(ValueError, "local import is missing"):
            self.parts(sources)
        # Reusing cached lexical facts must re-resolve the new reader's
        # missing/available module, even after a failed lookup at this path.
        sources[helper] = b"def numerical_value(value):\n    return float(value)\n"
        self.assertEqual(canonical_json_sha256(projected), canonical_json_sha256(self.parts(sources)))

    def test_cached_reader_bytes_and_returned_projection_remain_isolated(self):
        owner = "core/order_lot_policy.py"
        original = self.original[owner]
        changed = original.replace(b"return price_to_milli(limit_price) * qty_int",
                                   b"return price_to_milli(limit_price) + qty_int")
        self.assertEqual(len(original), len(changed))
        self.assertNotEqual(original, changed)
        first = self.parts({})
        first[owner] = "caller-local mutation"
        del first["core/native_optimizer_hotpath.c"]
        alternate_reader = self.parts({owner: changed})
        self.assertNotEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(alternate_reader))
        restored_reader = self.parts({owner: original})
        self.assertEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(restored_reader))

    def test_real_metrics_wrapper_and_new_reporting_helper_remain_numerical(self):
        owner = "services/audit/target_oracle_romd.py"
        original = self.original[owner]
        changed = original.replace(b"return oracle_observation(replay, target=target)",
            b"return {**oracle_observation(replay, target=target), 'romd': 999.0}")
        self.assertNotEqual(original, changed)
        self.assertNotEqual(canonical_json_sha256(self.baseline),
                            canonical_json_sha256(self.parts({owner: changed})))
        owner = "services/audit/oracle_observation.py"
        original = self.original[owner].decode("utf-8-sig")
        changed = original.replace('"romd": float(summary["return_over_max_drawdown"]),',
                                  '"romd": _new_metric(summary["return_over_max_drawdown"]),')
        changed += "\nfrom filters.breakout_quality.strategy_compare_reporting import new_metric as _new_metric\n"
        reporting = "filters/breakout_quality/strategy_compare_reporting.py"
        one = self.original[reporting] + b"\ndef new_metric(value):\n    return float(value) + 1\n"
        two = one.replace(b"return float(value) + 1", b"return float(value) + 2")
        self.assertNotEqual(canonical_json_sha256(self.parts({owner: changed.encode(), reporting: one})),
                            canonical_json_sha256(self.parts({owner: changed.encode(), reporting: two})))

    def test_observation_changes_and_non_docstring_literals_are_effective(self):
        owner = "services/audit/oracle_observation.py"
        original = self.original[owner].decode("utf-8-sig")
        changed = original.replace('"romd": float(summary["return_over_max_drawdown"]),',
            '"romd": float(summary["return_over_max_drawdown"]) + 1,')
        self.assertNotEqual(original, changed)
        self.assertNotEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(self.parts({owner: changed.encode()})))

    def test_unused_definition_defaults_and_decorators_preserve_import_time_effects(self):
        owner = "services/audit/oracle_observation.py"
        original = self.original[owner].decode("utf-8-sig")
        helper = "\ndef _eager_effect():\n    global numerical_adjustment\n    numerical_adjustment = 1\n    return lambda function: function\n"
        variants = (
            "\ndef unused_with_default(value=_eager_effect()):\n    return value\n",
            "\n@_eager_effect()\ndef unused_with_decorator(value):\n    return value\n",
            "\nclass UnusedClass:\n    active_effect = _eager_effect()\n",
        )
        for definition in variants:
            with self.subTest(definition=definition.splitlines()[1]):
                one = (original + helper + definition).encode()
                two = one.replace(b"numerical_adjustment = 1", b"numerical_adjustment = 2")
                self.assertNotEqual(canonical_json_sha256(self.parts({owner: one})),
                                    canonical_json_sha256(self.parts({owner: two})))
        changed = original.replace('    summary = _scenario_summary(replay)',
                                   '    summary = _scenario_summary(replay)\n    "non-docstring marker"')
        self.assertNotEqual(canonical_json_sha256(self.baseline), canonical_json_sha256(self.parts({owner: changed.encode()})))
