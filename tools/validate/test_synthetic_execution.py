"""Execution-only regressions; fixtures never use market/model artifacts."""
from __future__ import annotations

import importlib.util
import ast
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validate import validator_execution as execution


def _entry(fn):
    return SimpleNamespace(name=fn.__name__, validator=fn, layer="meta_contract", cost_class="fast")


def _first(params):
    return _rendezvous(params, "first", "second")


def _second(params):
    return _rendezvous(params, "second", "first")


def _rendezvous(params, own, other):
    root = Path(params["directory"])
    (root / own).write_text(str(os.getpid()))
    deadline = time.monotonic() + 5
    while not (root / other).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("validators did not overlap")
        time.sleep(0.01)
    params["child_mutation"] = own
    return [{"status": "PASS", "metric": own, "pid": os.getpid(),
             "threads": os.environ["OMP_NUM_THREADS"]}], {"ticker": own}


def _pass(params):
    return [{"status": "PASS", "metric": params}], {"ticker": "pass"}


def _fail(_params):
    return [{"status": "FAIL", "metric": "injected"}], {"ticker": "fail"}


def _raise(_params):
    raise ValueError("injected validator fault")


def _exit(_params):
    os._exit(7)


def _hang(_params):
    time.sleep(30)


class SyntheticExecutionContractTests(unittest.TestCase):
    def test_source_index_keeps_small_facts_and_invalidates_changes(self):
        from tools.validate import source_index as index
        source = ("import os.path\nfrom math import sqrt as root\nVALUE = 1\n"
                  "class C:\n    pass\nasync def async_probe():\n    pass\n"
                  "def validate_probe():\n    try:\n        import json\n"
                  "    except ValueError as exc:\n        raise\n")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"source.py"
            path.write_text(source)
            tree = ast.parse(source)
            self.assertEqual(index.read_source_top_level_bindings(path), {
                "exports": frozenset({"path", "root", "VALUE", "C", "async_probe", "validate_probe"}),
                "functions": ("validate_probe",)})
            self.assertEqual([ast.dump(n) for n in index.read_source_import_nodes(path)],
                             [ast.dump(n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))])
            self.assertEqual([ast.dump(n) for n in index.read_source_exception_handlers(path)],
                             [ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)])
            self.assertNotIn(path.resolve(), index._AST_CACHE)
            full = index.read_source_ast(path)
            self.assertIs(full, index.read_source_ast(path))
            path.write_text("def validate_changed():\n    pass\n")
            self.assertEqual(index.read_source_top_level_bindings(path)["functions"], ("validate_changed",))
            self.assertEqual(index.read_source_import_nodes(path), ())
            self.assertEqual(index.read_source_exception_handlers(path), ())
            self.assertNotIn(path.resolve(), index._AST_CACHE)
            self.assertIsNot(full, index.read_source_ast(path))
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                index.read_source_top_level_bindings(path)

    def test_planner_keeps_owners_together_and_all_cases_once(self):
        specs = [{"name": f"case_{i}", "module": module}
                 for i, module in enumerate(("a", "b", "a", "c", "b", "a", "d"))]
        batches = execution.plan_batches(specs, 3)
        self.assertEqual(sorted(i for batch in batches for i, _ in batch), list(range(len(specs))))
        for module in ("a", "b", "c", "d"):
            self.assertEqual(sum(any(s["module"] == module for _, s in batch) for batch in batches), 1)
        self.assertTrue(all([i for i, _ in b] == sorted(i for i, _ in b) for b in batches))

    def test_resource_bounds_and_invalid_budget(self):
        for cpus, expected in ((1, 1), (3, 1), (4, 1), (8, 4), (64, 4)):
            self.assertEqual(execution.worker_count(4, 10, cpu_count=cpus), expected)
        self.assertEqual(execution.worker_count(8, 2, cpu_count=64), 2)
        self.assertEqual(execution.worker_count(4, 20, cpu_count=8, threads=2), 1)
        for bad in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                execution.worker_count(bad, 10)

    def test_thread_budget_restores_loaded_torch_and_environment_on_error(self):
        state = [7]
        fake_torch = SimpleNamespace(get_num_threads=lambda: state[0],
                                    set_num_threads=lambda n: state.__setitem__(0, n))
        before = os.environ.copy()
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "fault"):
                with execution._worker_thread_budget(2):
                    self.assertEqual(state[0], 2)
                    self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
                    raise ValueError("fault")
        self.assertEqual(state[0], 7)
        self.assertEqual(dict(os.environ), before)

    def test_serial_checks_keep_failures_and_exception_identity(self):
        rows, summaries = execution.run_entries("input", [_entry(_pass), _entry(_fail)])
        self.assertEqual([row["status"] for row in rows], ["PASS", "FAIL"])
        self.assertEqual([s["validator_name"] for s in summaries], ["_pass", "_fail"])
        self.assertTrue(all(s["duration_sec"] >= 0 for s in summaries))
        with self.assertRaisesRegex(RuntimeError, "_raise: ValueError: injected"):
            execution.run_entries(None, [_entry(_raise)])

    def test_registry_descriptors_reject_duplicate_names(self):
        with self.assertRaises(ValueError):
            execution.describe_entries([_entry(_pass), _entry(_pass)])
        self.assertEqual(execution.describe_entries([_entry(_pass)])[0]["module"], __name__)

    def test_receipts_restore_order_and_reject_missing_duplicate_wrong_identity(self):
        specs = execution.describe_entries([_entry(_pass), _entry(_fail)])
        receipts = [(1, [{"status": "FAIL"}], {"validator_name": "_fail"}),
                    (0, [{"status": "PASS"}], {"validator_name": "_pass"})]
        self.assertEqual([r["status"] for r in execution.assemble_results(specs, receipts)[0]], ["PASS", "FAIL"])
        for invalid in (receipts[:1], receipts + receipts[:1],
                        [(9, [], {"validator_name": "_pass"})],
                        [(0, [], {"validator_name": "wrong"})]):
            with self.assertRaises(RuntimeError):
                execution.assemble_results(specs, invalid)

    def test_coverage_merge_checks_each_file_and_preserves_errors(self):
        imported = []
        class Data:
            def __init__(self, basename):
                self.path = basename
            def read(self):
                imported.append(self.path)
                if Path(self.path).read_text() == "corrupt":
                    raise ValueError("corrupt coverage")
            def has_arcs(self):
                return Path(self.path).read_text() == "arcs"
        combined = []
        cov = SimpleNamespace(get_data=lambda: SimpleNamespace(update=lambda d: combined.append(d.path)))
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {"coverage": SimpleNamespace(CoverageData=Data)}):
            a, b = Path(temp)/"a", Path(temp)/"b"
            a.write_text("arcs"); b.write_text("arcs")
            execution.merge_coverage(cov, [str(a), str(b)])
            self.assertEqual(imported, [str(a), str(b)])
            self.assertEqual(combined, imported)
            b.write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                execution.merge_coverage(cov, [str(b)])
            b.write_text("lines")
            with self.assertRaisesRegex(RuntimeError, "branch coverage"):
                execution.merge_coverage(cov, [str(b)])
            b.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing"):
                execution.merge_coverage(cov, [str(b)])

    def test_batch_branch_collection_saves_even_when_validator_crashes(self):
        calls = []
        class Coverage:
            def __init__(self, **kwargs): calls.append(kwargs)
            def start(self): calls.append("start")
            def stop(self): calls.append("stop")
            def save(self): calls.append("save")
        specs = execution.describe_entries([_entry(_raise)])
        with patch.dict(sys.modules, {"coverage": SimpleNamespace(Coverage=Coverage)}):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                execution._run_batch(None, list(enumerate(specs)), {"branch": True}, "isolated", 1)
        self.assertEqual(calls, [{"data_file": "isolated", "branch": True}, "start", "stop", "save"])

    def test_coverage_wrapper_uses_config_and_invalidates_previous_report(self):
        launcher = importlib.import_module("tools.validate.main")
        from tools.local_regression.common import load_manifest
        manifest = {**load_manifest(), "synthetic_workers": 2, "test_cpu_threads": 1}
        calls = []
        class Coverage:
            def __init__(self, **kwargs): self.path = Path(kwargs["data_file"])
            def get_option(self, name): return None
            def start(self): calls.append("start")
            def stop(self): calls.append("stop")
            def save(self): self.path.write_text("fixture"); calls.append("save")
            def json_report(self, *, outfile, pretty_print):
                Path(outfile).write_text("{}"); calls.append("json")
        entries = [_entry(_pass), _entry(_fail)]
        expected = ([{"status": "PASS"}, {"status": "FAIL"}],
                    [{"validator_name": "_pass"}, {"validator_name": "_fail"}])
        with tempfile.TemporaryDirectory() as temp, \
             patch.dict(sys.modules, {"coverage": SimpleNamespace(Coverage=Coverage)}), \
             patch("tools.local_regression.common.load_manifest", return_value=manifest), \
             patch.object(execution.os, "cpu_count", return_value=8), \
             patch.object(execution, "merge_coverage", side_effect=lambda *a: calls.append("merge")), \
             patch.object(execution, "run_parallel", return_value=(*expected, [])) as runner:
            got = launcher._run_synthetic_suite_with_optional_coverage(
                temp, None, lambda _: self.fail("unexpected serial fallback"),
                (RuntimeError,), validator_entries=entries)
            self.assertEqual(got, expected)
            self.assertEqual(runner.call_args.kwargs["workers"], 2)
            self.assertEqual(calls, ["start", "stop", "merge", "save", "json"])
            info = json.loads((Path(temp)/"coverage_artifacts/coverage_run_info.json").read_text())
            self.assertEqual(info["returncode"], 1)  # The failing assertion survives.
            self.assertEqual(info["synthetic_case_count"], 2)
            report = Path(temp)/"coverage_artifacts/coverage_synthetic.json"
            report.write_text('{"old_pass":true}')
            runner.side_effect = RuntimeError("worker failed")
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                launcher._run_synthetic_suite_with_optional_coverage(
                    temp, None, lambda _: self.fail("unexpected serial retry"),
                    (RuntimeError,), validator_entries=entries)
            self.assertNotIn("old_pass", report.read_text())
            info = json.loads((report.parent/"coverage_run_info.json").read_text())
            self.assertFalse(info["suite_completed"])
            self.assertEqual(info["returncode"], 1)
            manifest["synthetic_workers"] = 1
            runner.side_effect = AssertionError("serial setting must not spawn")
            self.assertEqual(launcher._run_synthetic_suite_with_optional_coverage(
                temp, None, lambda _: expected, (RuntimeError,), validator_entries=entries), expected)
            info = json.loads((report.parent/"coverage_run_info.json").read_text())
            self.assertEqual(info["synthetic_workers"], 1)

    def test_serial_config_keeps_original_runner(self):
        launcher = importlib.import_module("tools.validate.main")
        from tools.local_regression.common import load_manifest
        # No run directory is the original direct, coverage-free path.
        expected = ([{"status": "PASS"}], [{"validator_name": "fixture"}])
        self.assertEqual(launcher._run_synthetic_suite_with_optional_coverage(
            "", None, lambda _: expected, (RuntimeError,)), expected)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"manifest.json"
            path.write_text(json.dumps({"synthetic_workers": 1}))
            self.assertEqual(load_manifest(path)["synthetic_workers"], 1)
            for invalid in (0, -1, True, "4"):
                path.write_text(json.dumps({"synthetic_workers": invalid}))
                with self.assertRaises(RuntimeError):
                    load_manifest(path)

    @unittest.skipUnless(importlib.util.find_spec("coverage"), "coverage is not installed")
    def test_real_coverage_union_keeps_both_sides_of_branch(self):
        from coverage import Coverage, CoverageData
        with tempfile.TemporaryDirectory() as temp:
            paths = []
            filename = str(Path(temp) / "fixture.py")
            for index, arcs in enumerate(({(-1, 1), (1, 2), (2, -1)}, {(-1, 1), (1, 3), (3, -1)})):
                path = str(Path(temp)/f".coverage.synthetic.{index}")
                data = CoverageData(basename=path)
                data.add_arcs({filename: arcs}); data.write(); paths.append(path)
            cov = Coverage(data_file=str(Path(temp)/"combined"), branch=True)
            execution.merge_coverage(cov, paths)
            self.assertEqual(set(cov.get_data().arcs(filename)),
                             {(-1, 1), (1, 2), (2, -1), (1, 3), (3, -1)})


class SyntheticExecutionSpawnTests(unittest.TestCase):
    """Actual process lifecycle probes, also usable independently of formal tests."""
    def test_spawn_overlap_order_environment_and_parent_isolation(self):
        parent_env = os.environ.get("OMP_NUM_THREADS")
        with tempfile.TemporaryDirectory() as temp:
            params = {"directory": temp}
            specs = execution.describe_entries([_entry(_first), _entry(_second)])
            for spec in specs:
                module = f"_execution_fixture{spec['name']}"
                (Path(temp)/f"{module}.py").write_text(
                    f"from tools.validate.test_synthetic_execution import {spec['name']}\n")
                spec["module"] = module
            with patch.object(sys, "path", [temp, *sys.path]):
                rows, summaries, files = execution.run_parallel(params, specs,
                    workers=2, timeout=15, threads=2)
            self.assertEqual([r["metric"] for r in rows], ["first", "second"])
            self.assertEqual(len({r["pid"] for r in rows}), 2)
            self.assertEqual([r["threads"] for r in rows], ["2", "2"])
            self.assertNotIn("child_mutation", params)
            self.assertEqual([s["validator_name"] for s in summaries], ["_first", "_second"])
            self.assertEqual(files, [])
        self.assertEqual(os.environ.get("OMP_NUM_THREADS"), parent_env)

    def test_spawn_failure_exit_timeout_leave_no_children(self):
        original = {p.pid for p in multiprocessing.active_children()}
        for fn, error, timeout in ((_raise, RuntimeError, 10), (_exit, RuntimeError, 10), (_hang, TimeoutError, 0.3)):
            with self.subTest(fn=fn.__name__), self.assertRaises(error):
                execution.run_parallel(None, execution.describe_entries([_entry(fn)]),
                                       workers=1, timeout=timeout, threads=1)
            self.assertEqual({p.pid for p in multiprocessing.active_children()}, original)

    @unittest.skipUnless(importlib.util.find_spec("coverage"), "coverage is not installed")
    def test_actual_worker_coverage_matches_serial_branch_union(self):
        from coverage import Coverage
        with tempfile.TemporaryDirectory() as temp:
            paths, entries = [], []
            with patch.object(sys, "path", [temp, *sys.path]):
                for index in range(2):
                    module = f"_coverage_execution_fixture_{index}"
                    path = Path(temp)/f"{module}.py"
                    condition = "value" if index == 0 else "not value"
                    path.write_text(f"def probe_{index}(value):\n    if {condition}:\n        status = 'PASS'\n    else:\n        status = 'FAIL'\n    return [{{'status': status}}], {{}}\n")
                    paths.append(str(path))
                    fn = getattr(importlib.import_module(module), f"probe_{index}")
                    self.addCleanup(sys.modules.pop, module, None)
                    entries.append(SimpleNamespace(name=f"probe_{index}", validator=fn,
                                                    layer="meta_contract", cost_class="fast"))
                serial = Coverage(data_file=str(Path(temp)/"serial"), branch=True, include=paths)
                serial.start()
                expected = execution.run_entries(True, entries)
                serial.stop(); serial.save()
                rows, summaries, files = execution.run_parallel(True, execution.describe_entries(entries),
                    workers=2, timeout=15, threads=1, coverage_options={"branch": True, "include": paths}, coverage_dir=temp)
                merged = Coverage(data_file=str(Path(temp)/"merged"), branch=True, include=paths)
                execution.merge_coverage(merged, files)
                self.assertEqual(rows, expected[0])
                self.assertEqual([s["validator_name"] for s in summaries], [s["validator_name"] for s in expected[1]])
                for path in paths:
                    self.assertEqual(set(merged.get_data().arcs(path)), set(serial.get_data().arcs(path)))


if __name__ == "__main__":
    unittest.main()
