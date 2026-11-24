"""Behavioral bundle lifecycle checks using only temporary source snapshots."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import ast
from copy import deepcopy
from hashlib import sha256
import io
import json
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from apps import package_zip, run_bundle
from services.research.oracle_build_attestation import ATTESTATION_RELATIVE_PATH, EVIDENCE_DIRECTORY, OracleBuildBlockedError
from tools.local_regression.formal_pipeline import FORMAL_STEP_ORDER


class RunBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="run_bundle_behavior_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.module = self.root / "module.py"
        self.module.write_text("VALUE = 1\n")
        self.package = self.root / "source.zip"
        self.events = []
        self.formal_rc = 0
        self.formal_effect = lambda: None
        self.commit = "a" * 40
        self.candidate = {"source_inventory": {}, "source_digest": "b" * 64,
                          "component_revisions": {"replay": 1}, "equivalence_evidence": {},
                          "evidence_files": {}, "evidence_artifacts": {}}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for obj, name, value in ((run_bundle, "REPO_ROOT", self.root), (package_zip, "PROJECT_ROOT", self.root)):
            self.stack.enter_context(patch.object(obj, name, value))
        self.stack.enter_context(patch.object(run_bundle, "_git_head", return_value=self.commit))
        self.stack.enter_context(patch.object(run_bundle, "warn_if_experiment_log_staged"))
        self.stack.enter_context(patch.object(run_bundle, "run_cmd", side_effect=self.command))
        self.stack.enter_context(patch.object(package_zip, "create_package_snapshot", side_effect=self.package_snapshot))
        self.prepare = self.stack.enter_context(patch.object(run_bundle, "prepare_oracle_build_candidate", return_value=self.candidate))
        self.finalize = self.stack.enter_context(patch.object(run_bundle, "finalize_oracle_build_attestation", side_effect=self.certificate))
        self.publish = self.stack.enter_context(patch.object(run_bundle, "publish_oracle_build_attestation", side_effect=lambda *a, **kw: self.events.append("publish")))
        self.inspect = self.stack.enter_context(patch.object(run_bundle, "require_oracle_build_attestation", side_effect=lambda *a: self.events.append("inspect")))

    def package_snapshot(self):
        self.events.append("package")
        with zipfile.ZipFile(self.package, "w") as archive:
            archive.write(self.module, "module.py")
        self.original_zip = self.package.read_bytes()
        return package_zip.freeze_package_snapshot(self.package)

    def command(self, cmd, name, **kwargs):
        if cmd[:2] == ["git", "commit"]:
            self.events.append("commit")
        elif cmd[:3] == ["git", "diff", "--cached"]:
            return SimpleNamespace(returncode=1)
        elif "apps/test_suite.py" in cmd:
            self.events.append("test")
            self.formal_effect()
            summary = {"overall_status": "PASS" if self.formal_rc == 0 else "FAIL",
                       "git_commit": self.commit[:8], "failures": self.formal_rc,
                       "selected_steps": list(FORMAL_STEP_ORDER),
                       "scripts": [{"name": name, "status": "PASS"} for name in FORMAL_STEP_ORDER]}
            with zipfile.ZipFile(self.root / "to_chatgpt_bundle_test.zip", "w") as archive:
                archive.writestr("master_summary.json", json.dumps(summary))
            return SimpleNamespace(returncode=self.formal_rc)
        return SimpleNamespace(returncode=0)

    def certificate(self, root, **kwargs):
        self.events.append("certify")
        self.assertEqual(kwargs["formal_summary"]["git_commit"], self.commit[:8])
        return {"schema": "test-certificate", "evidence_artifacts": dict(self.candidate["evidence_artifacts"])}

    def invoke(self):
        with redirect_stdout(io.StringIO()):
            return run_bundle.main([])

    def test_commit_package_formal_then_finalize_and_readonly_inspection(self):
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.events, ["commit", "package", "test", "certify", "publish", "inspect"])
        with zipfile.ZipFile(self.package) as archive:
            self.assertIn(ATTESTATION_RELATIVE_PATH, archive.namelist())
            self.assertEqual(archive.read("module.py"), self.module.read_bytes())
            self.assertFalse(any("evidence/formal_bundle_" in name for name in archive.namelist()))
            delivery = json.loads(archive.read(package_zip.ORACLE_DELIVERY_MANIFEST))
            self.assertEqual(delivery["certificate_sha256"], sha256(archive.read(ATTESTATION_RELATIVE_PATH)).hexdigest())
        sidecar = self.root / delivery["archive"]
        self.assertEqual(sha256(sidecar.read_bytes()).hexdigest(), delivery["sha256"])
        with tarfile.open(sidecar, "r:xz") as archive:
            self.assertTrue(any("evidence/formal_bundle_" in name for name in archive.getnames()))

    def test_formal_failure_retains_commit_and_original_package(self):
        self.formal_rc = 3
        self.assertEqual(self.invoke(), 3)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.assertEqual(self.events, ["commit", "package", "test"])
        self.publish.assert_not_called()

    def test_unknown_source_proof_does_not_become_verified_from_formal_pass(self):
        self.prepare.side_effect = OracleBuildBlockedError("review required")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.finalize.assert_not_called()
        self.publish.assert_not_called()

    def test_source_changed_during_formal_rejects_finalization(self):
        self.formal_effect = lambda: self.module.write_text("VALUE = 2\n")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()

    def test_finalization_does_not_recollect_new_working_tree_files(self):
        self.formal_effect = lambda: (self.root / "late.py").write_text("not part of tested package\n")
        self.assertEqual(self.invoke(), 0)
        with zipfile.ZipFile(self.package) as archive:
            self.assertNotIn("late.py", archive.namelist())

    def test_finalization_preserves_level_nine_compression_for_frozen_members(self):
        content = b"".join(f"# record={index % 193:03d} group={index % 31:02d} value={index % 47:02d}\n".encode()
                           for index in range(20000))
        self.module.write_bytes(content)
        with zipfile.ZipFile(self.package, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            archive.write(self.module, "module.py")
        with zipfile.ZipFile(self.package) as archive:
            compressed_size = archive.getinfo("module.py").compress_size
        default_level = io.BytesIO()
        with zipfile.ZipFile(default_level, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr("module.py", content)
        with zipfile.ZipFile(default_level) as archive:
            self.assertGreater(archive.getinfo("module.py").compress_size, compressed_size)
        snapshot = package_zip.freeze_package_snapshot(self.package)
        package_zip.finalize_frozen_package(snapshot, attestation={"evidence_artifacts": {}})
        with zipfile.ZipFile(self.package) as archive:
            self.assertEqual(archive.read("module.py"), content)
            self.assertEqual(archive.getinfo("module.py").compress_size, compressed_size)

    def test_finalization_io_failure_returns_nonzero_without_publishing(self):
        with patch.object(package_zip, "atomic_replace_with_retry", side_effect=OSError("busy")):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()
        self.assertFalse(self.package.with_suffix(".zip.certifying").exists())
        self.assertFalse(self.package.with_suffix(".oracle-evidence.tar.xz.certifying").exists())

    def test_primary_replace_failure_retains_frozen_package_and_complete_evidence(self):
        original_replace = package_zip.atomic_replace_with_retry
        def fail_primary(source, destination, **kwargs):
            if Path(destination) == self.package:
                raise OSError("primary package busy")
            return original_replace(source, destination, **kwargs)
        with patch.object(package_zip, "atomic_replace_with_retry", side_effect=fail_primary):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()
        sidecars = list(self.root.glob("*.oracle-evidence-*.tar.xz"))
        self.assertEqual(len(sidecars), 1)
        with tarfile.open(sidecars[0], "r:xz") as archive:
            for name, record in self.candidate["evidence_files"].items():
                self.assertEqual(archive.extractfile(name).read(), Path(record["path"]).read_bytes())
        self.assertFalse(self.package.with_suffix(".zip.certifying").exists())
        self.assertFalse(self.package.with_suffix(".oracle-evidence.tar.xz.certifying").exists())

    def test_solid_evidence_archive_shares_repeated_bytes_without_removing_records(self):
        payload = b"".join(sha256(str(index).encode()).hexdigest().encode() + b"\n" for index in range(6000))
        names = [f"{EVIDENCE_DIRECTORY}/baseline_parity.json", f"{EVIDENCE_DIRECTORY}/current_parity.json"]
        contents = {names[0]: b"baseline\r\n" + payload, names[1]: b"current\n" + payload}
        archive_path = self.root / "evidence.tar.xz"
        digest = package_zip._write_oracle_evidence_archive(archive_path, contents)
        separate = io.BytesIO()
        with zipfile.ZipFile(separate, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        self.assertLess(archive_path.stat().st_size, len(separate.getvalue()) * 0.75)
        self.assertEqual(digest, sha256(archive_path.read_bytes()).hexdigest())
        with tarfile.open(archive_path, "r:xz") as archive:
            self.assertEqual(set(archive.getnames()), set(contents))
            for name, data in contents.items():
                self.assertEqual(archive.extractfile(name).read(), data)

    def test_archive_moves_matching_evidence_without_collecting_sidecars_as_source(self):
        self.package_snapshot()
        sidecar = self.root / f"source.oracle-evidence-{'a' * 16}.tar.xz"
        sidecar.write_bytes(b"complete evidence retained")
        unrelated = self.root / f"other.oracle-evidence-{'b' * 16}.tar.xz"
        unrelated.write_bytes(b"another delivery")
        self.assertTrue(package_zip._should_skip(Path(sidecar.name)))
        self.assertFalse(package_zip._should_skip(Path("user_data.tar.xz")))
        count, _ = package_zip._archive_existing_root_package_zips()
        self.assertEqual(count, 1)
        self.assertEqual((self.root / "arch" / sidecar.name).read_bytes(), b"complete evidence retained")
        self.assertFalse(sidecar.exists())
        self.assertTrue(unrelated.exists())

    def test_ambiguous_formal_evidence_is_rejected(self):
        self.formal_effect = lambda: (self.root / "to_chatgpt_bundle_other.zip").write_bytes(b"ambiguous")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()

    def test_tampered_zip_rejected_even_when_source_is_unchanged(self):
        self.formal_effect = lambda: self.package.write_bytes(self.package.read_bytes() + b"changed")
        self.assertEqual(self.invoke(), 1)
        self.publish.assert_not_called()

    def test_package_rejects_accidental_baseline_or_accumulated_formal_evidence(self):
        snapshot = self.package_snapshot()
        for names, message in (([f"{EVIDENCE_DIRECTORY}/baseline_sources_{'a' * 16}.zip"], "external audit"),
                               ([f"{EVIDENCE_DIRECTORY}/formal_bundle_{'a' * 16}.zip",
                                 f"{EVIDENCE_DIRECTORY}/formal_bundle_{'b' * 16}.zip"], "only the current formal")):
            with self.subTest(names=names):
                records = {name: {"path": "not_read", "sha256": "a" * 64} for name in names}
                certificate = {"evidence_artifacts": {name: record["sha256"] for name, record in records.items()}}
                with self.assertRaisesRegex(RuntimeError, message):
                    package_zip.finalize_frozen_package(snapshot, attestation=certificate, evidence_files=records)
                self.assertEqual(self.package.read_bytes(), self.original_zip)

    def orchestration_tree(self):
        tree = ast.parse(Path(run_bundle.__file__).read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        package_index = next(index for index, node in enumerate(main.body)
                             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Attribute)
                             and node.value.func.attr == "create_package_snapshot")
        test_index = next(index for index, node in enumerate(main.body)
                          if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                          and isinstance(node.value.func, ast.Name) and node.value.func.id == "run_cmd"
                          and any(isinstance(child, ast.Constant) and child.value == "apps/test_suite.py"
                                  for child in ast.walk(node.value)))
        return tree, main, package_index, test_index

    def test_shared_entry_checker_accepts_frozen_api_and_ignores_console_text(self):
        from tools.validate.meta_contracts import summarize_bundle_orchestration_ast
        tree, main, _, _ = self.orchestration_tree()
        self.assertTrue(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
                node.args = [ast.Constant(value="A different progress message")]
        self.assertTrue(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])

    def test_shared_entry_checker_rejects_reversed_or_missing_package(self):
        from tools.validate.meta_contracts import summarize_bundle_orchestration_ast
        tree, main, package, formal = self.orchestration_tree()
        main.body[package], main.body[formal] = main.body[formal], main.body[package]
        self.assertFalse(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])
        tree, main, package, _ = self.orchestration_tree()
        del main.body[package]
        self.assertFalse(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])

    def test_shared_entry_checker_rejects_conditional_dead_package_and_duplicate_tests(self):
        from tools.validate.meta_contracts import summarize_bundle_orchestration_ast
        tree, main, package, _ = self.orchestration_tree()
        main.body[package] = ast.If(test=ast.Constant(value=False), body=[main.body[package]], orelse=[])
        self.assertFalse(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])
        tree, main, package, _ = self.orchestration_tree()
        hidden = ast.parse("def never_called():\n    pass\n").body[0]
        hidden.body = [main.body.pop(package)]
        main.body.insert(package, hidden)
        self.assertFalse(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])
        tree, main, _, formal = self.orchestration_tree()
        main.body.insert(0, deepcopy(main.body[formal]))
        contract = summarize_bundle_orchestration_ast(tree)
        self.assertFalse(contract["bundle_invokes_test_suite"])
        self.assertFalse(contract["bundle_packages_before_tests"])

    def test_shared_entry_checker_requires_canonical_package_binding(self):
        from tools.validate.meta_contracts import summarize_bundle_orchestration_ast
        tree, _, _, _ = self.orchestration_tree()
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "apps" and any(alias.name == "package_zip" for alias in node.names):
                node.module = "unrelated"
        self.assertFalse(summarize_bundle_orchestration_ast(tree)["bundle_packages_before_tests"])


if __name__ == "__main__":
    unittest.main()
