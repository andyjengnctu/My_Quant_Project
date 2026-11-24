"""Source-inspection invariants, independent of market/model artifacts."""
from __future__ import annotations

import ast
import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validate import source_index as index
from tools.validate import transient_code_maintenance as maintenance


class _IsolatedSourceCacheTest(unittest.TestCase):
    def setUp(self):
        # These probes run inside the metadata validator too. Cache invalidation
        # fixtures must not discard the enclosing worker's real source index.
        for name in ('_TEXT_CACHE', '_AST_CACHE', '_STRUCTURE_CACHE'):
            cache = patch.object(index, name, {})
            cache.start()
            self.addCleanup(cache.stop)


class SourceInspectionContractTests(_IsolatedSourceCacheTest):
    def test_import_ownership_accepts_equivalent_syntax_but_rejects_shadow_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "consumer.py"
            for statement in (
                "from core.walk_forward_policy import filter_search_train_dates",
                "from core.walk_forward_policy import (\n    filter_search_train_dates,\n)",
                "from core.walk_forward_policy import filter_search_train_dates as filter_dates",
            ):
                path.write_text(statement, encoding="utf-8")
                self.assertTrue(index.imports_symbol_from(path, "core.walk_forward_policy", "filter_search_train_dates"))
            for statement in (
                '# from core.walk_forward_policy import filter_search_train_dates\n',
                'TEXT = "from core.walk_forward_policy import filter_search_train_dates"',
                "from another_owner import filter_search_train_dates",
                "from core.walk_forward_policy import different_function",
            ):
                path.write_text(statement, encoding="utf-8")
                self.assertFalse(index.imports_symbol_from(path, "core.walk_forward_policy", "filter_search_train_dates"))

    def test_compact_structure_matches_ast_for_nested_language_constructs(self):
        source = '''import os as operating
from . import child
@decorate("on" if mode == "MR-71" else "off")
async def probe(value=(1 if mode == "C72" else 2)):
    async with context():
        async for item in items:
            try:
                from pkg import one
            except ValueError:
                import fallback
            else:
                from pkg import two
            finally:
                import finalizer
    while mode == ("S" "R-73"):
        break
    match value:
        case {"kind": "C74"} if (True if mode == "MR-75" else False):
            import matched
    return ["yes" if mode == "MR-76" else "no" for item in items]
class Plain:
    pass
'''
        tree = ast.parse(source)
        expected_branches = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Match):
                tests = [case.pattern for case in node.cases]
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                tests = [node.test]
            else:
                continue
            expected_branches.append((node.lineno, tuple(
                child.value for test in tests for child in ast.walk(test)
                if isinstance(child, ast.Constant) and isinstance(child.value, str))))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"probe.py"
            path.write_text(source)
            with patch.object(index.ast, "parse", wraps=ast.parse) as parse:
                self.assertEqual(index.read_source_branch_literals(path), tuple(expected_branches))
                self.assertEqual([ast.dump(n) for n in index.read_source_import_nodes(path)],
                                 [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
                self.assertEqual([ast.dump(n) for n in index.read_source_exception_handlers(path)],
                                 [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)])
                definitions = index.read_source_definitions(path)
                self.assertEqual([(d.name, d.kind, d.decorated) for d in definitions],
                                 [("probe", "AsyncFunctionDef", True), ("Plain", "ClassDef", False)])
                self.assertEqual(index.read_source_top_level_function_names(path), ("probe",))
                self.assertEqual(parse.call_count, 1)
            self.assertNotIn(path.resolve(), index._AST_CACHE)
            index.clear_source_index_cache()
            # The faster statements-only path must discover the same imports.
            self.assertEqual([ast.dump(n) for n in index.read_source_import_nodes(path)],
                             [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])

    def test_definition_and_branch_facts_refresh_on_edit_and_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"probe.py"
            path.write_text('def before():\n    if label == "MR-7":\n        return 1\n')
            self.assertEqual(index.read_source_definitions(path)[0].name, "before")
            self.assertEqual(index.read_source_branch_literals(path), ((2, ("MR-7",)),))
            path.write_text('async def after():\n    return "new"\n')
            self.assertEqual(index.read_source_branch_literals(path), ())
            self.assertEqual(index.read_source_definitions(path)[0].kind, "AsyncFunctionDef")
            self.assertEqual(index.read_source_definitions(path)[0].name, "after")
            path.unlink()
            for reader in (index.read_source_branch_literals, index.read_source_definitions):
                with self.assertRaises(FileNotFoundError):
                    reader(path)

    def test_gc_state_restored_for_success_syntax_error_and_caller_disabled(self):
        original_gc = gc.isenabled()
        original_parse = ast.parse
        states = []
        def observe_parse(*args, **kwargs):
            states.append(gc.isenabled())
            return original_parse(*args, **kwargs)
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/"probe.py"
                for enabled in (True, False):
                    gc.enable() if enabled else gc.disable()
                    with patch.object(index.ast, "parse", side_effect=observe_parse):
                        path.write_text("def valid():\n    pass\n")
                        index.clear_source_index_cache()
                        self.assertEqual(index.read_source_definitions(path)[0].name, "valid")
                        self.assertEqual(gc.isenabled(), enabled)
                        self.assertNotIn(path.resolve(), index._AST_CACHE)
                        index.read_source_ast(path, retain=False)
                        self.assertNotIn(path.resolve(), index._AST_CACHE)
                        self.assertEqual(gc.isenabled(), enabled)
                        path.write_text("if broken\n")
                        with self.assertRaises(SyntaxError):
                            index.read_source_definitions(path)
                        self.assertEqual(gc.isenabled(), enabled)
            self.assertTrue(states)
            self.assertFalse(any(states))
        finally:
            gc.enable() if original_gc else gc.disable()


class MaintenanceInspectionContractTests(_IsolatedSourceCacheTest):
    def test_branch_prefilter_preserves_decoded_and_concatenated_literals(self):
        spellings = [
            '"MR-71"', '"M" "R-71"', '("M" # split\n "R-71")',
            '"M" r"R-71"', '"M" f"R-71"', '"\\x4dR-71"',
            "'''M''' \"R-71\"", '\"M\" """R-71"""',
            '("""M"""\n """R-71""")',
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'core/probe.py'
            path.parent.mkdir()
            for spelling in spellings:
                with self.subTest(spelling=spelling):
                    path.write_text('if mode == '+spelling+':\n    pass\n')
                    _graph, findings = maintenance._analyze_static_python_tree(root, {'core.probe': path})
                    self.assertEqual([r['identities'] for r in findings], [['MR-71']])
            path.write_text('if value > 0:\n    result = value\n')
            with patch.object(maintenance, 'read_source_branch_literals', side_effect=AssertionError('unneeded expression walk')):
                _graph, findings = maintenance._analyze_static_python_tree(root, {'core.probe': path})
            self.assertEqual(findings, [])

    def test_advisory_findings_preserve_reachability_literals_and_private_definitions(self):
        sources = {
            "core/main.py": 'import tools.audit.live\nif mode == ("M" "R-71"):\n    pass\n'
                            'value = "a" if mode == "\\x53R-72" else "b"\n'
                            'match mode:\n    case "C73":\n        pass\n'
                            'def _unused():\n    pass\n'
                            '@decorate\ndef _decorated():\n    pass\n'
                            'def _referenced():\n    pass\n_ref = _referenced\n'
                            'def __excluded():\n    pass\n',
            "tools/audit/live.py": 'VALUE = 1\n',
            "tools/audit/unused.py": 'VALUE = 2\n',
            "tools/audit/alias.py": 'import sys\nsys.modules[__name__] = _impl\n',
            "tools/validate/synthetic_fixture_cases.py":
                'def validate_fixture(path):\n    return path.read_text()\n',
            "doc/TEST_SUITE_CHECKLIST.md": '# fixture\n',
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, text in sources.items():
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            with patch.object(maintenance, "_active_dynamic_audit_modules", return_value=set()), \
                 patch.object(maintenance, "_catalog_modes_by_module", return_value={}), \
                 patch.object(maintenance, "_git_parent_available", return_value=False):
                result = maintenance.summarize_transient_code_maintenance(root)
                self.assertEqual([r["module"] for r in result["stale_modules"]], ["tools.audit.unused"])
                self.assertEqual([r["name"] for r in result["private_zero_callers"]], ["_unused"])
                self.assertEqual({v for r in result["experiment_execution_branches"] for v in r["identities"]},
                                 {"MR-71", "SR-72", "C73"})
                self.assertEqual(result["source_shape_inventory"], {"line_count": 2, "validator_count": 1})
                self.assertTrue(result["advisory_only"])
                # A later invocation must discover changed membership/content.
                (root/"tools/audit/unused.py").unlink()
                (root/"core/main.py").write_text('import tools.audit.live\n')
                updated = maintenance.summarize_transient_code_maintenance(root)
                self.assertEqual(updated["stale_modules"], [])
                self.assertEqual(updated["private_zero_callers"], [])
                self.assertEqual(updated["experiment_execution_branches"], [])

    def test_git_growth_keeps_previous_and_current_validator_semantics(self):
        old = 'def validate_old(path):\n    return path.read_text()\n'
        current = 'async def validate_mr99_probe(path):\n    return path.read_text()\n'
        relative = 'tools/validate/synthetic_fixture_cases.py'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/relative
            path.parent.mkdir(parents=True)
            path.write_text(current)
            with patch.object(maintenance, '_git_parent_available', return_value=True), \
                 patch.object(maintenance, '_git_text', return_value=old), \
                 patch.object(maintenance.subprocess, 'run', return_value=SimpleNamespace(
                     returncode=0, stdout='2\t2\t'+relative+'\n')):
                result = maintenance._git_growth_summary(root, maintenance._source_shape_validator_inventory(root))
            self.assertTrue(result['available'])
            self.assertEqual(result['source_shape_delta'], 0)
            self.assertEqual(result['new_experiment_validators'],
                             [{'path': relative, 'validator': 'validate_mr99_probe'}])
            self.assertIn('new_experiment_specific_validator', result['review_reasons'])


if __name__ == '__main__':
    unittest.main()
