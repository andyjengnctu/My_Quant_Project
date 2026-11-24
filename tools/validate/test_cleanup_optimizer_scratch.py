"""Only superseded, outcome-free Optuna work may expire; real studies are fixtures."""
from datetime import datetime, timezone
from contextlib import closing
from hashlib import sha256
import io
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import optuna

from services.cleanup_optimizer_scratch import OptimizerScratchCleanup
from services.optimizer.runtime import _OptimizerStudyWriteLock
from services.optimizer.strategy_param_training import (
    _build_selection_full_roos_schedule_contract, _selection_full_roos_runtime_contract,
    _selection_full_roos_optimizer_runtime_cache_identity,
)


def _hold_study_writer(path, ready, release):
    lock = _OptimizerStudyWriteLock(path)
    try:
        ready.set()
        release.wait(15)
    finally:
        lock.release()


class OptimizerScratchCleanupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "outputs/optimizer/strategy_param_schedule/canonical/full/schedule_fixture"
        self.db_dir = self.workspace / "optimizer_runtime/outer_rolling_oos/db"
        self.db_dir.mkdir(parents=True)
        schedule = _build_selection_full_roos_schedule_contract(
            first_oos_date="2021-01-01", last_oos_date="2026-03-02",
            train_window_months=120, oos_months=12,
        )
        args = SimpleNamespace(dataset="full", trials_per_fold=300, fixed_risk=0.01,
                               max_position_cap_pct=0.3, max_positions=10, rotation="off")
        self.contract = _selection_full_roos_runtime_contract(
            root=self.root, args=args, schedule_contract=schedule,
            dataset_identity_override={"schema_version": 1, "fixture_source": "independent"},
        )
        self.current = self.contract["runtime_identity_sha256"]
        self.preflight = self.workspace / "rolling_preflight.json"
        self.preflight.write_text(json.dumps({
            **self.contract, "status": "P4_HISTORY_PREFLIGHT_PASS",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }), encoding="utf-8")
        self.old = self.db_path("a" * 64)
        self.new = self.db_path(self.current)
        self.create_study(self.old, "a" * 64, completed=False)
        self.create_study(self.new, self.current, completed=True)
        os.utime(self.old, (1, 1))
        self.relative = self.old.relative_to(self.root)
        self.changes = []
        self.worker = SimpleNamespace(
            apply=True,
            lifecycle=SimpleNamespace(protected=lambda _: False, removed=self.changes.append),
        )

    def db_path(self, identity):
        return self.db_dir / ("outer_oos_runtime_" + identity[:20] + "_202101.db")

    def create_study(self, path, identity, *, completed):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        storage = optuna.storages.RDBStorage("sqlite:///" + path.as_posix())
        try:
            study = optuna.create_study(storage=storage)
            study.set_user_attr("optimizer_runtime_cache_identity", identity)
            trial = study.ask()
            trial.suggest_float("atr_times_init", 1.0, 4.5, step=0.1)
            trial.set_user_attr("fixed_tp_percent", 0.0)
            if completed:
                study.tell(trial, 0.25)
        finally:
            storage.remove_session()
            storage.engine.dispose()

    def prepared(self):
        adapter = OptimizerScratchCleanup(self.root)
        adapter.prepare({self.relative: self.old.stat().st_size})
        return adapter

    def execute(self, statement, *, path=None):
        with closing(sqlite3.connect(path or self.old)) as db:
            with db:
                db.execute(statement)
        os.utime(self.old, (1, 1))

    def lifecycle(self, adapter):
        from services.cleanup_lifecycle import LifecycleCleanup
        from services.project_cleaner import walk
        scope = LifecycleCleanup(self.root)
        for relative, size in walk(self.root, lambda *args: None, scope.observe_directory):
            scope.observe(relative, size)
        scope.extra_targets.update(adapter.reference_targets)
        scope.extra_roots.update(adapter.retained_roots)
        scope.prepare(set())
        return scope

    def test_superseded_result_free_study_and_read_only_preview(self):
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.root.rglob("*")}
        adapter = self.prepared()
        self.assertTrue(adapter.candidate(self.relative))
        self.worker.apply = False
        with adapter.deletion_lease(self.worker, self.relative):
            self.assertTrue(adapter.proof(self.worker, self.relative))
        self.assertEqual(before, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.root.rglob("*")})
        self.worker.apply = True
        with adapter.deletion_lease(self.worker, self.relative):
            self.assertTrue(adapter.proof(self.worker, self.relative))
            adapter.verify_before_delete(self.worker, self.relative)
        self.assertTrue(self.old.exists() and self.new.exists())
        self.assertEqual(adapter.sources, {})
        self.assertIn(self.new.relative_to(self.root), adapter.retained_roots)

    def test_both_current_cache_identity_forms_are_kept(self):
        identities = {self.current, _selection_full_roos_optimizer_runtime_cache_identity(
            self.contract, optimizer_search_protocol="fixed_multiseed_v1",
        )}
        for identity in identities:
            with self.subTest(identity=identity):
                target = self.db_path(identity)
                if target.exists():
                    target.unlink()
                self.create_study(target, identity, completed=False)
                os.utime(target, (1, 1))
                adapter = OptimizerScratchCleanup(self.root)
                relative = target.relative_to(self.root)
                adapter.prepare({relative: target.stat().st_size})
                self.assertFalse(adapter.candidate(relative))

    def test_evidence_snapshot_does_not_refresh_changed_inputs(self):
        from core.file_integrity import file_change_signature
        second = self.db_path("b" * 64)
        self.create_study(second, "b" * 64, completed=False)
        os.utime(second, (1, 1))
        adapter = OptimizerScratchCleanup(self.root)
        adapter.prepare({self.relative: self.old.stat().st_size,
                         second.relative_to(self.root): second.stat().st_size})
        observed = adapter.proof_revisions
        self.assertTrue({self.old, second, self.new, self.preflight} <= observed.keys())
        self.assertEqual(adapter.proof_scopes, {self.workspace.relative_to(self.root)})
        self.assertEqual(observed[Path(str(second) + "-journal")], ("missing",))
        copy = adapter.proof_revisions
        copy.clear()
        self.preflight.write_bytes(self.preflight.read_bytes() + b" ")
        self.assertNotEqual(observed[self.preflight], file_change_signature(self.preflight))
        self.assertEqual(adapter.proof_revisions, observed)
        with adapter.deletion_lease(self.worker, self.relative):
            with self.assertRaises(ValueError):
                adapter.proof(self.worker, self.relative)

    def test_any_terminal_or_waiting_trial_is_preserved(self):
        for state in ("COMPLETE", "FAIL", "PRUNED", "WAITING"):
            with self.subTest(state=state):
                self.execute("UPDATE trials SET state='" + state + "'")
                self.assertFalse(self.prepared().candidate(self.relative))

    def test_result_values_heartbeats_and_unknown_diagnostics_are_preserved(self):
        statements = (
            "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES (1, 0, 0.5, 'FINITE')",
            "INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value, intermediate_value_type) VALUES (1, 0, 0.5, 'FINITE')",
            "INSERT INTO trial_heartbeats (trial_id, heartbeat) VALUES (1, '2020-01-01 00:00:00')",
            "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (1, 'failure_evidence', '\"keep\"')",
            "INSERT INTO study_system_attributes (study_id, key, value_json) VALUES (1, 'sampler_state', '{}')",
            "INSERT INTO study_user_attributes (study_id, key, value_json) VALUES (1, 'outcome_summary', '{}')",
        )
        for statement in statements:
            with self.subTest(statement=statement):
                self.execute(statement)
                self.assertFalse(self.prepared().candidate(self.relative))
                table = statement.split()[2]
                condition = {"trial_user_attributes": " WHERE key='failure_evidence'",
                             "study_user_attributes": " WHERE key='outcome_summary'"}.get(table, "")
                self.execute("DELETE FROM " + table + condition)

    def test_sqlite_sidecars_block_inspection_and_late_publication(self):
        for suffix in ("-journal", "-wal", "-shm"):
            with self.subTest(suffix=suffix):
                adapter = self.prepared()
                sidecar = Path(str(self.old) + suffix)
                sidecar.write_bytes(b"busy")
                self.assertFalse(self.prepared().candidate(self.relative))
                with adapter.deletion_lease(self.worker, self.relative):
                    with self.assertRaises(ValueError):
                        adapter.proof(self.worker, self.relative)
                sidecar.unlink()

    def test_new_preflight_requires_canonical_producer_content_and_valid_signature(self):
        original = self.preflight.read_text(encoding="utf-8")
        for key, value in (("runtime_identity_sha256", "f" * 64), ("mode", "unrecognized_producer")):
            with self.subTest(key=key):
                payload = json.loads(original)
                payload[key] = value
                self.preflight.write_text(json.dumps(payload), encoding="utf-8")
                self.assertFalse(self.prepared().candidate(self.relative))
        self.preflight.write_text(original, encoding="utf-8")
        os.utime(self.preflight, (0, 0))
        self.assertFalse(self.prepared().candidate(self.relative))

    def test_missing_or_ambiguous_successor_never_authorizes_cleanup(self):
        self.new.unlink()
        self.assertFalse(self.prepared().candidate(self.relative))
        self.create_study(self.new, self.current, completed=True)
        other = _selection_full_roos_optimizer_runtime_cache_identity(
            self.contract, optimizer_search_protocol="fixed_multiseed_v1",
        )
        self.create_study(self.db_path(other), other, completed=True)
        self.assertFalse(self.prepared().candidate(self.relative))

    def test_malformed_preflight_and_signed_unknown_contract_are_preserved(self):
        from core.file_integrity import canonical_json_sha256
        original = self.preflight.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["unknown_scientific_contract"] = True
        payload["runtime_identity_sha256"] = canonical_json_sha256({
            key: value for key, value in payload.items()
            if key not in {"created_at", "status", "runtime_identity_sha256"}
        })
        for value in ([], {"rolling_policy": None}, payload):
            with self.subTest(value=value):
                self.preflight.write_text(json.dumps(value), encoding="utf-8")
                self.assertFalse(self.prepared().candidate(self.relative))

    def test_journals_and_new_writer_also_protect_the_retained_successor(self):
        adapter = self.prepared()
        journal = Path(str(self.new) + "-journal")
        journal.write_bytes(b"busy")
        self.assertFalse(self.prepared().candidate(self.relative))
        journal.unlink()
        lock = _OptimizerStudyWriteLock(str(self.new))
        try:
            with self.assertRaisesRegex(RuntimeError, "OPTIMIZER_STUDY_BUSY"):
                with adapter.deletion_lease(self.worker, self.relative):
                    self.fail("active successor writer lease was ignored")
        finally:
            lock.release()
        with adapter.deletion_lease(self.worker, self.relative):
            self.assertTrue(adapter.proof(self.worker, self.relative))

    def test_final_revalidation_detects_a_result_added_after_proof(self):
        adapter = self.prepared()
        with adapter.deletion_lease(self.worker, self.relative):
            self.assertTrue(adapter.proof(self.worker, self.relative))
            self.execute("UPDATE trials SET state='COMPLETE'")
            with self.assertRaises(ValueError):
                adapter.verify_before_delete(self.worker, self.relative)

    def test_final_revalidation_detects_changed_preflight_or_retained_study(self):
        for target in (self.preflight, self.new):
            with self.subTest(target=target):
                adapter = self.prepared()
                original = target.read_bytes()
                try:
                    with adapter.deletion_lease(self.worker, self.relative):
                        self.assertTrue(adapter.proof(self.worker, self.relative))
                        target.write_bytes(original + b" ")
                        with self.assertRaises(ValueError):
                            adapter.verify_before_delete(self.worker, self.relative)
                finally:
                    target.write_bytes(original)

    def test_exact_or_workspace_reference_preserves_the_scratch(self):
        adapter = self.prepared()
        reference = self.root / "state/consumer.json"
        reference.parent.mkdir()
        for target in (self.relative, self.relative.parent, self.workspace.relative_to(self.root)):
            with self.subTest(target=target):
                reference.write_text(json.dumps({"requires": target.as_posix()}))
                self.worker.lifecycle = self.lifecycle(adapter)
                self.assertTrue(self.worker.lifecycle.protected(self.relative))
                self.worker.apply = False
                with adapter.deletion_lease(self.worker, self.relative):
                    with self.assertRaises(ValueError):
                        adapter.proof(self.worker, self.relative)

    def test_real_writer_in_another_process_blocks_the_cleanup_lease(self):
        adapter = self.prepared()
        context = multiprocessing.get_context("spawn")
        ready, release = context.Event(), context.Event()
        worker = context.Process(target=_hold_study_writer, args=(str(self.old), ready, release))
        worker.start()
        try:
            self.assertTrue(ready.wait(10))
            with self.assertRaisesRegex(RuntimeError, "OPTIMIZER_STUDY_BUSY"):
                with adapter.deletion_lease(self.worker, self.relative):
                    self.fail("active producer lease was ignored")
            self.assertTrue(self.old.exists())
        finally:
            release.set()
            worker.join(10)
            if worker.is_alive():
                worker.terminate()
                worker.join(10)

    def test_cleaner_preview_and_apply_delete_sequential_scratch_in_one_workspace(self):
        from contextlib import redirect_stdout
        from core.file_integrity import file_change_signature
        from services.project_cleaner import Cleaner

        subprocess.run(["git", "-C", str(self.root), "init", "-q"], check=True,
                       capture_output=True)
        # The application creates its report directory before inventory.
        (self.root / "outputs/project_cleanup").mkdir()
        second_old = self.old.with_name(self.old.name.replace("_202101.db", "_202201.db"))
        second_new = self.new.with_name(self.new.name.replace("_202101.db", "_202201.db"))
        self.create_study(second_old, "a" * 64, completed=False)
        self.create_study(second_new, self.current, completed=True)
        os.utime(second_old, (1, 1))
        old_paths = {self.old, second_old}
        old_relatives = {path.relative_to(self.root) for path in old_paths}
        preserved = {path: sha256(path.read_bytes()).hexdigest()
                     for path in (self.new, second_new, self.preflight)}
        before = {path: file_change_signature(path) for path in self.root.rglob("*")}

        for apply in (False, True):
            with self.subTest(apply=apply):
                rows = []
                report = SimpleNamespace(row=lambda *args, **kwargs: rows.append(args))
                with redirect_stdout(io.StringIO()):
                    cleaner = Cleaner(self.root, report, apply=apply)
                    cleaner.scan()
                    self.assertEqual({relative for relative, _ in
                                      cleaner.storage_candidates["optimizer_scratch"]}, old_relatives)
                    cleaner.clear_unused()
                action = "DELETE" if apply else "PLAN_DELETE"
                self.assertEqual({row[1] for row in rows if row[0] == action}, old_relatives, rows)
                self.assertFalse([row for row in rows if row[0] in {"SKIP", "ERROR"}], rows)
                self.assertEqual(preserved, {path: sha256(path.read_bytes()).hexdigest()
                                             for path in preserved})
                self.assertTrue(all(path.exists() != apply for path in old_paths))
                if not apply:
                    self.assertEqual(before, {path: file_change_signature(path)
                                              for path in self.root.rglob("*")})
                    continue
                # Both deletions and their persistent owner locks must leave
                # the inventory usable for the next lifecycle scope.
                with cleaner.lifecycle.session(cleaner):
                    pass
                for path in (*old_paths, self.new, second_new):
                    self.assertEqual(Path(str(path) + ".writer.lock").read_bytes(), b"\0")
                    lock = _OptimizerStudyWriteLock(str(path))
                    lock.release()


if __name__ == "__main__":
    unittest.main()
