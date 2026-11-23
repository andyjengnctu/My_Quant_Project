"""Regression checks for destructive boundaries of the standalone cleaner."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import py_compile
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from app import cleaner
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
