"""Behavioral bundle lifecycle checks using only temporary source snapshots."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from apps import package_zip, run_bundle
from services.research.oracle_build_attestation import ATTESTATION_RELATIVE_PATH, OracleBuildBlockedError
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
            self.assertTrue(any("evidence/formal_bundle_" in name for name in archive.namelist()))

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

    def test_finalization_io_failure_returns_nonzero_without_publishing(self):
        with patch.object(package_zip, "atomic_replace_with_retry", side_effect=OSError("busy")):
            self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()
        self.assertFalse(self.package.with_suffix(".zip.certifying").exists())

    def test_ambiguous_formal_evidence_is_rejected(self):
        self.formal_effect = lambda: (self.root / "to_chatgpt_bundle_other.zip").write_bytes(b"ambiguous")
        self.assertEqual(self.invoke(), 1)
        self.assertEqual(self.package.read_bytes(), self.original_zip)
        self.publish.assert_not_called()

    def test_tampered_zip_rejected_even_when_source_is_unchanged(self):
        self.formal_effect = lambda: self.package.write_bytes(self.package.read_bytes() + b"changed")
        self.assertEqual(self.invoke(), 1)
        self.publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
