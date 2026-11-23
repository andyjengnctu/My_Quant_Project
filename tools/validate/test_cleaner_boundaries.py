"""Physical project boundaries and nonrecursive cache-directory cleanup."""
import os
from pathlib import Path
import py_compile
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from services.project_cleaner import retained
from tools.validate import test_project_cleaner as existing


class CleanerBoundaryTest(unittest.TestCase):
    setUp = existing.CleanupTest.setUp
    git = existing.CleanupTest.git
    write = existing.CleanupTest.write
    menu = existing.CleanupTest.menu
    scanned_cleaner = existing.CleanupTest.scanned_cleaner

    def test_positive_proof_cannot_authorize_absolute_or_traversal_paths(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside) / "only-copy.txt"
            external.write_bytes(b"irreplaceable")
            local = self.write("only-copy.txt", external.read_bytes())
            worker, rows = self.scanned_cleaner()
            invalid = [external, local, Path("../") / Path(outside).name / external.name,
                       Path("services/../only-copy.txt"), Path("only-copy.txt:stream"), Path(".")]
            with patch.object(worker, "proof", return_value="synthetic disposal proof"):
                for rel in invalid:
                    worker.remove_one("fixture", rel, len(b"irreplaceable"), None)
            self.assertEqual(external.read_bytes(), b"irreplaceable")
            self.assertEqual(local.read_bytes(), b"irreplaceable")
            self.assertEqual(sum(row[0] == "SKIP" for row in rows), len(invalid))

    def test_only_emptied_proven_cache_directories_are_removed(self):
        source = self.write("services/sample.py", b"value = 1\n")
        other = self.write("tools/sample.py", b"value = 2\n")
        first = Path(py_compile.compile(str(source), doraise=True))
        second = Path(py_compile.compile(str(other), doraise=True))
        unknown = self.write(second.parent / "notes.txt", b"keep")
        empty = self.root / "core/__pycache__"
        empty.mkdir(parents=True)
        self.menu(["1", "0"])
        self.assertTrue(first.exists() and second.exists())
        self.menu(["2", "0"])
        self.assertFalse(first.parent.exists())
        self.assertFalse(second.exists())
        self.assertEqual(unknown.read_bytes(), b"keep")
        self.assertTrue(empty.is_dir())
        self.menu(["2", "0"])
        self.assertTrue(empty.is_dir() and unknown.exists())

    def test_windows_case_variants_keep_protected_domains(self):
        for name in ("DATA/raw.csv", "STATE/account.json", ".GIT/index",
                     "Outputs/Trading/accounts/private.txt", "Outputs/PROJECT_CLEANUP/last_apply.zip"):
            with self.subTest(name=name):
                self.assertEqual(retained(Path(name), set())[0], "KEEP")

    @unittest.skipUnless(os.name == "nt", "Windows junction boundary")
    def test_junction_never_allows_deletion_outside_project(self):
        with tempfile.TemporaryDirectory() as outside:
            original = Path(outside) / "only-copy.txt"
            original.write_bytes(b"keep outside")
            junction = self.root / "linked"
            created = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), outside],
                                     capture_output=True, check=False)
            self.assertEqual(created.returncode, 0, created.stderr)
            try:
                worker, rows = self.scanned_cleaner()
                with patch.object(worker, "proof", return_value="synthetic disposal proof"):
                    worker.remove_one("fixture", Path("linked/only-copy.txt"), 12, None)
                self.assertEqual(original.read_bytes(), b"keep outside")
                self.assertTrue(any(row[0] == "SKIP" for row in rows))
            finally:
                # Nonrecursive removal of this test's junction, never its target.
                junction.rmdir()


if __name__ == "__main__":
    unittest.main()
