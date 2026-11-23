"""Regression checks for destructive boundaries of the standalone cleaner."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import py_compile
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from apps import cleaner
from core.compare_csv_storage import copy_reused_detail, publish_detail_csv
from core.regression_fixture_access import fixture_access
from services import cleanup_rules as rules, project_cleaner as service
from services.research.strategy_compare_pair_integrity import (
    build_completed_pair_file_integrity, completed_strategy_run_publication_matches,
)


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.write("doc/PROJECT_SETTINGS.md", b"test")
        self.git("init", "-q")
        self.base = Path("outputs/strategy_compare/preview/rolling/runs")
        self.output = "score_ranking_orderable_candidates.csv"
        self.payload = b"name,value\n" + b"test,123\n" * 1000

    def git(self, *args):
        subprocess.run(["git", "-C", str(self.root), *args], capture_output=True, check=True)

    def write(self, rel, data):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return path

    def publish(self, run, *, pinned=False, data=None):
        rel = self.base / run
        pair = rel / "pairs/arm"
        value = {"metadata": {"name": "arm"}}
        self.write(pair / "strategy_comparison.json", value)
        self.write(pair / "strategy_comparison.md", b"md")
        p = self.write(pair / self.output, self.payload if data is None else data)
        os.utime(p, ns=(1710000000000000000, 1710000000000000000))
        info = {"status": "COMPLETED", "config_fingerprint": "0123456789ab", "pairs": {"arm": value}}
        manifest = {"config_fingerprint": info["config_fingerprint"], "run_dir": rel.as_posix()}
        if pinned:
            record = build_completed_pair_file_integrity({"arm": p.parent})
            info["pair_file_integrity"] = manifest["pair_file_integrity"] = record
        self.write(rel / "strategy_comparison.json", info)
        self.write(rel / "manifest.json", manifest)
        return p

    def menu(self, answers):
        text = io.StringIO()
        with patch.object(cleaner, "ROOT", self.root), patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(text):
            cleaner.main()
        return text.getvalue()

    def summary(self, mode):
        with ZipFile(self.root / "outputs/project_cleanup" / f"last_{mode}.zip") as result:
            self.assertIsNone(result.testzip())
            return json.loads(result.read("summary.json")), list(csv.DictReader(
                io.StringIO(result.read("decisions.csv").decode("utf-8-sig"))))

    def scanned_cleaner(self):
        rows = []
        worker = service.Cleaner(self.root, SimpleNamespace(row=lambda *args, **kwargs: rows.append(args)), True)
        with contextlib.redirect_stdout(io.StringIO()):
            worker.scan()
        return worker, rows

    def test_newly_tracked_candidate_is_preserved(self):
        a, b, c = [self.publish(name) for name in ("a", "b", "c")]
        worker, rows = self.scanned_cleaner()
        self.git("add", str(b))
        worker.clear_unused()
        worker.share()
        self.assertFalse(a.samefile(b))
        self.assertTrue(a.samefile(c))
        self.assertTrue(any(row[0] == "KEEP" and row[1] == b.relative_to(self.root) for row in rows))

    def test_git_changes_during_hashing_are_seen_before_replacement(self):
        a, b = self.publish("a"), self.publish("b")
        worker, rows = self.scanned_cleaner()
        original_hash = worker.hash_file
        def stage_after_hash(path):
            digest = original_hash(path)
            if path == b:
                self.git("add", str(b))
            return digest
        with patch.object(worker, "hash_file", side_effect=stage_after_hash):
            worker.share()
        self.assertFalse(a.samefile(b))
        self.assertFalse(any(row[0] == "LINK" for row in rows))

    def test_missing_file_does_not_discard_other_candidates(self):
        a, b, c = [self.publish(name) for name in ("a", "b", "c")]
        worker, rows = self.scanned_cleaner()
        a.unlink()
        worker.clear_unused()
        worker.share()
        self.assertTrue(b.samefile(c))
        self.assertTrue(any(row[0] == "SKIP" and row[1] == a.relative_to(self.root) for row in rows))

    def test_updated_legacy_source_requires_fresh_migration_proof(self):
        rel = Path("models/run_best_params.json")
        source = self.write(rel, b"old")
        worker, rows = self.scanned_cleaner()
        worker.legacy = {rel: "content_sha_match"}
        source.write_bytes(b"new")
        worker.remove_one("legacy", rel, 3, set())
        self.assertEqual(source.read_bytes(), b"new")
        self.assertEqual(rows[-1][0], "KEEP")

    def test_interrupted_link_syscall_can_be_retried(self):
        a, b = self.publish("a"), self.publish("b")
        original_link = os.link
        def link_then_interrupt(src, dst):
            original_link(src, dst)
            raise KeyboardInterrupt
        with patch("services.project_cleaner.os.link", side_effect=link_then_interrupt):
            self.menu(["2", "0"])
        self.assertFalse((self.root / "outputs/project_cleanup/pending.link").exists())
        self.menu(["2", "0"])
        self.assertTrue(a.samefile(b))

    def test_failed_stage_removal_preserves_recovery_intent(self):
        a, b = self.publish("a"), self.publish("b")
        stage = self.root / "outputs/project_cleanup/pending.link"
        original_unlink, original_replace = Path.unlink, os.replace
        def reject_stage_unlink(path, *args, **kwargs):
            if path == stage:
                raise PermissionError("injected stage is busy")
            return original_unlink(path, *args, **kwargs)
        def reject_stage_replace(src, dst):
            if Path(src) == stage:
                raise PermissionError("injected destination is busy")
            return original_replace(src, dst)
        with patch.object(Path, "unlink", reject_stage_unlink), \
             patch("services.project_cleaner.os.replace", side_effect=reject_stage_replace):
            self.menu(["2", "0"])
        self.assertTrue(stage.exists())
        self.assertTrue((stage.parent / "apply.pending.csv").exists())
        self.assertIn("略過", self.summary("apply")[0]["status"])
        self.menu(["2", "0"])
        self.assertFalse(stage.exists())
        self.assertTrue(a.samefile(b))

    def test_malformed_pair_mapping_is_retained(self):
        a, b = self.publish("a"), self.publish("b")
        for p in (a, b):
            result = p.parents[2] / "strategy_comparison.json"
            payload = json.loads(result.read_text())
            payload["pairs"] = ["bad shape"]
            result.write_text(json.dumps(payload))
        self.menu(["2", "0"])
        self.assertFalse(a.samefile(b))

    def test_corrupt_history_returns_to_menu(self):
        target = self.root / "outputs/project_cleanup/last_apply.zip"
        target.parent.mkdir(parents=True)
        for contents in (None, {}, {"mode": "apply", "status": "已完成", "counts": []}):
            with self.subTest(contents=contents):
                with ZipFile(target, "w") as archive:
                    archive.writestr("decisions.csv", "")
                    if contents is not None:
                        archive.writestr("summary.json", json.dumps(contents))
                self.assertIn("本次操作未完成", self.menu(["3", "0"]))

    def test_cached_publication_rejects_later_changes(self):
        for changed in ("csv", "manifest", "pair_json"):
            with self.subTest(changed=changed):
                a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
                worker, rows = self.scanned_cleaner()
                for p in (a, b):
                    self.assertTrue(worker.published(p.relative_to(self.root)))
                    if changed == "csv":
                        stamp = p.stat().st_mtime_ns
                        p.write_bytes(b"!" * len(self.payload))
                        os.utime(p, ns=(stamp, stamp))
                    elif changed == "manifest":
                        (p.parents[2] / "manifest.json").write_text("{}")
                    else:
                        (p.parent / "strategy_comparison.json").write_text("{}")
                worker.share()
                self.assertFalse(a.samefile(b))
                self.assertFalse(any(row[0] == "LINK" for row in rows))

    def test_publication_and_dedup_read_each_csv_once(self):
        from core import file_integrity
        a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
        file_integrity._observed_file_sha256.cache_clear()
        with patch.object(file_integrity, "compute_file_sha256", wraps=file_integrity.compute_file_sha256) as hashed:
            self.menu(["1", "0"])
        observed = [Path(call.args[0]) for call in hashed.call_args_list]
        self.assertEqual(observed.count(a), 1)
        self.assertEqual(observed.count(b), 1)
        self.assertEqual(self.summary("preview")[0]["counts"]["PLAN_LINK"], 1)

    def test_menu_proofs_dedup_retry_and_independent_rewrites(self):
        a, b = self.publish("a", pinned=True), self.publish("b")
        unequal = self.publish("c", data=b"Z" * len(self.payload))
        partial = self.publish("d")
        (partial.parents[2] / "manifest.json").unlink()
        src = self.write("services/module.py", b"v=42")
        self.git("add", "services/module.py")
        pyc = Path(py_compile.compile(str(src), doraise=True))
        stale = self.write(self.base / "a/pairs/arm/strategy_comparison.html", b"retired")
        fixture = self.write(rules.STAGING / "quick_gate_broken_run_best_params_12345678/models/run_best_params.json",
                             b"{not-valid-json")
        custom = self.write(rules.STAGING / "quick_gate_broken_run_best_params_xxxxxxxx/models/run_best_params.json",
                            b"important")
        protected = [self.write(x, b"keep") for x in ("data/trading/raw", "state/trading/open",
                     "models/research/checkpoint.pt", "outputs/optimizer/study.db", "outputs/unclassified/draft")]
        before, report = self.menu(["", "0"]), self.summary("preview")[0]
        self.assertIn("預估可釋放", before)
        self.assertEqual(report["counts"]["PLAN_LINK"], 1)
        self.assertTrue(pyc.exists() and stale.exists() and fixture.exists())
        self.assertFalse(os.path.samefile(a, b))
        self.menu(["2", "3", self.output, "0"])
        stats, rows = self.summary("apply")
        self.assertEqual(stats["counts"]["LINK"], 1)
        self.assertEqual(stats["counts"]["DELETE"], 3)
        self.assertTrue(os.path.samefile(a, b))
        self.assertTrue(all(item.exists() for item in [custom, partial, unequal, *protected]))
        self.assertFalse(os.path.samefile(a, unequal))
        self.assertFalse(os.path.samefile(a, partial))
        self.assertFalse(pyc.exists() or stale.exists() or fixture.exists())
        for p in (a,b):
            directory=p.parents[2]
            self.assertTrue(completed_strategy_run_publication_matches(
                run_dir=directory, project_root=self.root,
                run_payload=json.loads((directory / "strategy_comparison.json").read_text())))
        self.menu(["2","0"])
        self.assertEqual(sum(self.summary("apply")[0]["saved"].values()), 0)
        replay = self.root / self.base / "e/pairs/arm" / self.output
        replay.parent.mkdir(parents=True)
        copy_reused_detail(a, replay)
        self.assertTrue(os.path.samefile(a,replay))
        import pandas as pd
        publish_detail_csv(pd.DataFrame({"new":[1]}), replay, index=False, encoding="utf-8-sig")
        self.assertFalse(os.path.samefile(a,replay))
        self.assertEqual(a.read_bytes(), self.payload)
        self.assertEqual(b.read_bytes(), self.payload)

    def test_reuse_copies_independently_and_traces_why_link_was_skipped(self):
        source = self.publish("original")
        for run, metadata, link_error, reason in (
            ("link_error", [True, True], OSError("link unavailable"), "link unavailable"),
            ("different_acl", [True, False], None, "不同檔案安全屬性"),
        ):
            with self.subTest(run=run):
                pair = self.root / self.base / run / "pairs/arm"
                dest = pair / source.name
                with patch("core.compare_csv_storage.file_metadata", side_effect=metadata), \
                     patch("core.compare_csv_storage.os.link", side_effect=link_error) as link, \
                     self.assertLogs("core.compare_csv_storage", level="DEBUG") as log:
                    shutil.copytree(source.parent, pair, copy_function=copy_reused_detail)
                self.assertEqual(link.call_count, int(link_error is not None))
                message, = log.output
                for fragment in (reason, source.relative_to(self.root).as_posix(), dest.relative_to(self.root).as_posix()):
                    self.assertIn(fragment, message)
                self.assertEqual(dest.read_bytes(), self.payload)
                self.assertEqual(dest.stat().st_mtime_ns, source.stat().st_mtime_ns)
                self.assertFalse(dest.samefile(source))
                dest.write_bytes(b"changed")
                self.assertEqual(source.read_bytes(), self.payload)

    def test_refuse_unmanaged_links_corruption_active_work_and_tracked(self):
        a, b = self.publish("a", pinned=True), self.publish("b", pinned=True)
        a.write_bytes(b"!" * len(self.payload))
        os.utime(a, ns=(b.stat().st_mtime_ns, b.stat().st_mtime_ns))
        self.write(self.base / "a/pairs/arm/strategy_comparison.html", b"retired")
        alias=self.root / "private.csv"
        os.link(b,alias)
        src=self.write("services/tracked.py", b"x=1")
        pyc=Path(py_compile.compile(str(src),doraise=True))
        self.git("add","-f",str(src),str(pyc))
        stub=self.write(rules.STAGING / "quick_gate_broken_run_best_params_12345678/models/run_best_params.json", b"{not-valid-json")
        with fixture_access(self.root / rules.STAGING, cleanup=False):
            self.menu(["2","0"])
        self.assertTrue(stub.is_file() and pyc.is_file() and (a.parent/"strategy_comparison.html").is_file())
        self.assertFalse(os.path.samefile(a,b))
        self.assertTrue(os.path.samefile(b,alias))
        self.assertEqual(self.summary("apply")[0]["counts"].get("LINK",0),0)

    def test_change_in_model_roster_preserves_cache_and_fit(self):
        import numpy as np
        from core.strategy_compare_policy import _compatibility_catalogs
        from config import breakout_quality
        slots = rules.retired_diagnostics(self.root)
        self.assertTrue(slots)
        rel, profile = next(iter(slots.items()))
        eligible = self.root / rel
        eligible.parent.mkdir(parents=True)
        fields = ("identity", "ticker", "dates", "labels", "raw_r", "valid", "payload_sha256")
        values = {key: np.array([0]) for key in fields}
        values.update(identity=np.asarray("a" * 64), payload_sha256=np.asarray("b" * 64))
        np.savez(eligible, **values)
        fit = self.write(Path("models/research") / profile / "model.pt", b"unique historical model")
        self.assertIn(profile, {v.get("experiment_profile") for v in _compatibility_catalogs()[1].values()})
        with patch.object(breakout_quality, "BREAKOUT_QUALITY_MODEL_RESEARCH_EXPERIMENT_PROFILE", profile):
            self.assertNotIn(rel, rules.retired_diagnostics(self.root))
            self.menu(["2", "0"])
            self.assertTrue(eligible.exists())
        self.menu(["2", "0"])
        self.assertFalse(eligible.exists())
        self.assertTrue(fit.is_file())

    def test_robust_work_requires_committed_evidence_and_lease(self):
        run = rules.ROBUST / "suite/rolling/1234567890abcdef"
        work = self.write(run / "work/results/result.json", b"temp")
        replay = self.write(run / "work/replay/seed_result.json", b"durable if policy requires")
        proof = self.write(run / "robustness_summary.json", b"permanent")
        from core.file_integrity import exclusive_file_lock, sibling_publication_lock_path
        lock_path = sibling_publication_lock_path(self.root / run)
        self.menu(["1", "0"])
        self.assertFalse(lock_path.exists())
        self.menu(["2", "0"])
        self.assertTrue(work.is_file())
        from services.research import strategy_multi_seed_robustness as owner
        with patch.object(owner, "_validate_completed_run", return_value={"contract": {"retention": {"keep_replay_details": True}}}):
            with exclusive_file_lock(sibling_publication_lock_path(self.root / run)):
                self.menu(["2", "0"])
                self.assertTrue(work.is_file())
            self.menu(["2", "0"])
        self.assertFalse(work.is_file())
        self.assertTrue(proof.is_file() and replay.is_file())

    def test_recover_abrupt_interruption_between_link_and_replace(self):
        a, b = self.publish("a"), self.publish("b")
        from services.cleanup_report import FIELDS
        target = self.root / "outputs/project_cleanup"
        target.mkdir(parents=True)
        with (target / "apply.pending.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(FIELDS)
            writer.writerow(["INTENT", b.relative_to(self.root).as_posix(), len(self.payload), 0, "link",
                             a.relative_to(self.root).as_posix()])
        os.link(a, target / "pending.link")
        self.menu(["3", "", "0"])
        self.assertFalse((target / "pending.link").exists())
        self.assertFalse(os.path.samefile(a,b))
        self.assertIn("中斷", self.summary("apply")[0]["status"])
        self.menu(["2","0"])
        self.assertTrue(os.path.samefile(a,b))

    def test_interrupt_and_recover(self):
        a,b=self.publish("a"),self.publish("b")
        with patch.object(service.Cleaner,"link_one",side_effect=KeyboardInterrupt):
            self.menu(["2","0"])
        self.assertIn("中斷",self.summary("apply")[0]["status"])
        self.assertFalse(os.path.samefile(a,b))
        self.menu(["2","0"])
        self.assertTrue(os.path.samefile(a,b))


if __name__ == "__main__":
    unittest.main()
