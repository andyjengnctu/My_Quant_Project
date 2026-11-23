"""Isolated checks for backup identity, failure handling and cleaner dispatch.

Robocopy is mocked: these checks do not copy or delete real project artifacts.
"""
from contextlib import redirect_stdout
from datetime import datetime
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import call, patch

from apps import cleaner
from services import project_backup as service


class ProjectBackupTest(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.base = Path(self.temp).resolve()
        self.root = self.base / "project"
        self.root.mkdir()
        (self.root / "doc").mkdir()
        (self.root / "doc/PROJECT_SETTINGS.md").write_text("fixture", encoding="utf-8")
        self.drive = self.base / "D"
        self.drive.mkdir()
        self.enterContext(patch.object(service.sys, "platform", "win32"))
        self.process = self.enterContext(patch.object(service.subprocess, "run"))
        self.codes(1, 0)
        self.output = io.StringIO()
        self.enterContext(redirect_stdout(self.output))

    def codes(self, *codes):
        self.process.reset_mock()
        self.process.side_effect = [SimpleNamespace(returncode=code) for code in codes]

    def backup(self, **kwargs):
        return service.backup_project(
            kwargs.get("root", self.root), backup_drive=kwargs.get("drive", self.drive)
        )

    def freeze_identity(self):
        clock = self.enterContext(patch.object(service, "datetime"))
        clock.now.return_value = datetime(2026, 11, 24, 15, 30, 12)
        unique = self.enterContext(patch.object(service, "uuid4"))
        unique.return_value = SimpleNamespace(hex="a7c92e01000000000000000000000000")
        return unique

    def test_allowed_copy_and_verify_codes_finish_in_order(self):
        for copy_code in range(8):
            for verify_code in (0, 2):
                with self.subTest(copy=copy_code, verify=verify_code):
                    self.codes(copy_code, verify_code)
                    target = self.backup()
                    self.assertTrue(target.is_dir())
                    first, second = self.process.call_args_list
                    self.assertEqual(first.args[0][:3], ["robocopy.exe", str(self.root), str(target)])
                    self.assertEqual(second.args[0][:3], first.args[0][:3])
                    self.assertNotIn("/L", first.args[0])
                    self.assertTrue({"/Z", "/MT:8", "/R:2", "/W:2"} <= set(first.args[0]))
                    self.assertTrue({"/L", "/R:0", "/W:0"} <= set(second.args[0]))
                    self.assertNotIn("/Z", second.args[0])
                    for invocation in (first, second):
                        self.assertTrue({"/E", "/COPY:DAT", "/DCOPY:DAT", "/NP", "/TEE"}
                                        <= set(invocation.args[0]))
                        self.assertFalse({"/256", "/MIR", "/PURGE", "/MOV", "/MOVE"}
                                         & set(invocation.args[0]))
                        self.assertIs(invocation.kwargs["check"], False)
                    self.assertIn("備份與唯讀比對完成", self.output.getvalue())

    def test_copy_failure_never_runs_verification_or_claims_success(self):
        for code in (8, 16, 24, -1, 3221225786):
            with self.subTest(code=code):
                self.codes(code)
                with self.assertRaisesRegex(RuntimeError, f"代碼 {code}"):
                    self.backup()
                self.process.assert_called_once()
        self.assertNotIn("備份與唯讀比對完成", self.output.getvalue())

    def test_verification_rejects_pending_copies_mismatches_and_errors(self):
        for code in (1, 3, 4, 5, 6, 7, 8, 16, -1, 3221225786):
            with self.subTest(code=code):
                self.codes(1, code)
                with self.assertRaisesRegex(RuntimeError, f"代碼 {code}"):
                    self.backup()
                self.assertEqual(self.process.call_count, 2)
        self.assertNotIn("備份與唯讀比對完成", self.output.getvalue())

    def test_each_run_has_a_separate_id_even_in_the_same_second(self):
        unique = self.freeze_identity()
        unique.side_effect = [SimpleNamespace(hex="a" * 32), SimpleNamespace(hex="b" * 32)]
        first = self.backup()
        self.codes(1, 0)
        second = self.backup()
        self.assertEqual(first.name, "My_Quant_Project_20261124_153012_aaaaaaaa")
        self.assertEqual(second.name, "My_Quant_Project_20261124_153012_bbbbbbbb")
        self.assertTrue(first.is_dir())
        self.assertTrue(second.is_dir())

    def test_folder_and_both_external_logs_share_the_same_id(self):
        self.freeze_identity()
        target = self.backup()
        identifier = "20261124_153012_a7c92e01"
        self.assertEqual(target.name, f"My_Quant_Project_{identifier}")
        self.assertIn(f"備份辨識碼：{identifier}", self.output.getvalue())
        for invocation, stage in zip(self.process.call_args_list, ("backup", "verify")):
            logs = [x.removeprefix("/UNILOG:") for x in invocation.args[0] if x.startswith("/UNILOG:")]
            self.assertEqual(logs, [str(self.drive / f"My_Quant_Project_{stage}_{identifier}.log")])
            self.assertFalse(Path(logs[0]).is_relative_to(target))
            self.assertFalse(Path(logs[0]).is_relative_to(self.root))

    def test_id_collision_preserves_prior_backup_and_runs_nothing(self):
        self.freeze_identity()
        target = self.drive / "My_Quant_Project_20261124_153012_a7c92e01"
        target.mkdir()
        sentinel = target / "valuable.txt"
        sentinel.write_bytes(b"previous backup")
        with self.assertRaisesRegex(RuntimeError, "辨識碼已存在"):
            self.backup()
        self.process.assert_not_called()
        self.assertEqual(sentinel.read_bytes(), b"previous backup")

    def test_original_and_legacy_backup_are_not_changed(self):
        source = self.root / "research.bin"
        source.write_bytes(b"source evidence")
        legacy = self.drive / "My_Quant_Project"
        legacy.mkdir()
        prior = legacy / "old.txt"
        prior.write_bytes(b"old backup")
        target = self.backup()
        self.assertNotEqual(target, legacy)
        self.assertEqual(source.read_bytes(), b"source evidence")
        self.assertEqual(prior.read_bytes(), b"old backup")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["doc", "research.bin"])

    def test_long_unicode_paths_and_spaces_are_passed_as_single_arguments(self):
        root = self.root / ("資料 & " + "a" * 90) / ("b" * 100) / ("c" * 100)
        root.mkdir(parents=True)
        drive = self.drive / "備份 空間 (D) & records"
        drive.mkdir()
        self.assertGreater(len(str(root)), 260)
        target = self.backup(root=root, drive=drive)
        for invocation in self.process.call_args_list:
            self.assertEqual(invocation.args[0][1:3], [str(root), str(target)])
            self.assertFalse(invocation.kwargs.get("shell", False))
            self.assertFalse(invocation.kwargs.get("capture_output", False))

    def test_missing_source_or_drive_and_relative_drive_stop_before_copy(self):
        for kwargs in ({"root": self.base / "missing"}, {"drive": self.base / "missing"},
                       {"drive": "relative"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.backup(**kwargs)
        file = self.drive / "not_a_drive"
        file.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "找不到備份磁碟"):
            self.backup(drive=file)
        self.process.assert_not_called()
        self.assertEqual(file.read_bytes(), b"keep")

    def test_copying_a_parent_into_itself_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "互相包含"):
            self.backup(drive=self.root)
        self.process.assert_not_called()
        self.assertEqual([p.name for p in self.root.iterdir()], ["doc"])

    def test_alias_into_source_is_rejected(self):
        alias = self.drive / "alias"
        try:
            alias.symlink_to(self.root, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "互相包含"):
            self.backup(drive=alias)
        self.process.assert_not_called()

    def test_non_windows_reports_requirement_without_starting_a_process(self):
        with patch.object(service.sys, "platform", "linux"):
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                self.backup()
        self.process.assert_not_called()
        self.assertEqual(list(self.drive.iterdir()), [])

    def test_process_start_errors_and_interruptions_keep_partial_backup(self):
        for stage in (0, 1):
            for error in (FileNotFoundError("robocopy.exe"), PermissionError("denied"), KeyboardInterrupt()):
                with self.subTest(stage=stage, error=type(error).__name__):
                    self.process.reset_mock()
                    self.process.side_effect = [SimpleNamespace(returncode=1)] * stage + [error]
                    before = set(self.drive.iterdir())
                    with self.assertRaisesRegex(RuntimeError, "備份尚未驗證完成"):
                        self.backup()
                    self.assertEqual(self.process.call_count, stage + 1)
                    self.assertEqual(len(set(self.drive.iterdir()) - before), 1)
        self.assertNotIn("備份與唯讀比對完成", self.output.getvalue())

    def test_menu_keeps_existing_choices_and_dispatches_backup(self):
        with patch.object(cleaner, "ROOT", self.root), patch.object(cleaner, "run") as cleanup, \
                patch.object(cleaner, "show_history") as history, \
                patch.object(cleaner, "backup_project") as backup, \
                patch("builtins.input", side_effect=["", "2", "3", "4", "9", "0"]):
            cleaner.main()
        self.assertEqual(cleanup.call_args_list, [call(self.root, apply=False), call(self.root, apply=True)])
        history.assert_called_once_with(self.root)
        backup.assert_called_once_with(self.root)
        self.assertIn("[4] 備份整個專案至 D 槽並驗證", self.output.getvalue())
        self.assertIn("請輸入 0～4", self.output.getvalue())

    def test_menu_backup_success_and_failure_never_dispatch_cleanup(self):
        for verify_code in (0, 1):
            with self.subTest(verify=verify_code):
                self.codes(1, verify_code)
                with patch.object(cleaner, "ROOT", self.root), patch.object(cleaner, "run") as cleanup, \
                        patch.object(cleaner, "backup_project", side_effect=lambda root: self.backup(root=root)), \
                        patch("builtins.input", side_effect=["4", "0"]):
                    cleaner.main()
                cleanup.assert_not_called()
                self.assertEqual(self.process.call_count, 2)
        self.assertIn("本次操作未完成", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
