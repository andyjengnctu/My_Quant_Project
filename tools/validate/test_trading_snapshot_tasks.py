"""Isolated concurrency contracts for Trading snapshots and GUI task delivery."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from core.file_integrity import load_json_strict
from services.trading import workbench_snapshot as snapshots
from services.trading.account_dashboard import build_trading_account_dashboard_read_model, publish_trading_account_dashboard_snapshot
from services.trading.account_state import initialize_trading_account_state, load_trading_account_state, set_trading_cash_balance
from services.trading.order_state import load_trading_order_state
from services.trading.pending_entry_state import load_trading_pending_entry_state
from services.trading.state_lock import TradingStateBusyError
from services.trading.state_snapshot import trading_snapshot_reads
from services.workbench_ui.background_tasks import BackgroundTaskRunner


class SnapshotTests(unittest.TestCase):
    def test_capture_blocks_mutation_then_projection_releases_lock(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_trading_account_state(root, cash=100)
            captured = threading.Event()
            release = threading.Event()
            original = snapshots.load_trading_account_state
            def paused_read(*args, **kwargs):
                state = original(*args, **kwargs)
                captured.set()
                self.assertTrue(release.wait(5))
                return state
            with ThreadPoolExecutor(max_workers=1) as pool, patch.object(snapshots, "load_trading_account_state", paused_read):
                future = pool.submit(snapshots.capture_trading_state_snapshot, root)
                self.assertTrue(captured.wait(5))
                try:
                    with self.assertRaises(TradingStateBusyError):
                        set_trading_cash_balance(root, cash=200, expected_revision=0)
                finally:
                    release.set()
                snapshot = future.result(timeout=5)
            set_trading_cash_balance(root, cash=200, expected_revision=0)
            with trading_snapshot_reads(snapshot):
                self.assertEqual(0, load_trading_account_state(root)["revision"])
                self.assertIsNone(load_trading_order_state(root))
                self.assertIsNone(load_trading_pending_entry_state(root))
                changed = load_trading_account_state(root)
                changed["revision"] = 999
                self.assertEqual(0, load_trading_account_state(root)["revision"])
                with self.assertRaisesRegex(RuntimeError, "projection"):
                    set_trading_cash_balance(root, cash=300, expected_revision=0)
            self.assertEqual(1, load_trading_account_state(root)["revision"])

    def test_accounting_projects_one_revision_while_live_state_advances(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_trading_account_state(root, cash=100)
            entered = threading.Event()
            release = threading.Event()
            def delayed_dashboard(project_root):
                entered.set()
                self.assertTrue(release.wait(5))
                return build_trading_account_dashboard_read_model(project_root)
            with ThreadPoolExecutor(max_workers=1) as pool, patch.object(snapshots, "build_trading_account_dashboard_read_model", delayed_dashboard):
                future = pool.submit(snapshots.build_trading_accounting_snapshot, root)
                self.assertTrue(entered.wait(5))
                try:
                    set_trading_cash_balance(root, cash=200, expected_revision=0)
                finally:
                    release.set()
                account, dashboard = future.result(timeout=5)
            self.assertEqual(0, account["revision"])
            self.assertEqual(account["revision"], dashboard["source_account_revision"])
            self.assertEqual(1, load_trading_account_state(root)["revision"])

    def test_late_old_publication_cannot_replace_new_revision(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            initialize_trading_account_state(root, cash=100)
            old = build_trading_account_dashboard_read_model(root)
            set_trading_cash_balance(root, cash=200, expected_revision=0)
            new = build_trading_account_dashboard_read_model(root)
            newer_done = threading.Event()
            def publish_old():
                self.assertTrue(newer_done.wait(5))
                return publish_trading_account_dashboard_snapshot(root, old)
            with ThreadPoolExecutor(max_workers=1) as pool:
                old_future = pool.submit(publish_old)
                paths = publish_trading_account_dashboard_snapshot(root, new)
                newer_done.set()
                self.assertEqual(paths, old_future.result(timeout=5))
            for relative in paths.values():
                self.assertEqual(1, load_json_strict(root / relative)["source_account_revision"])


class FakeScheduler:
    def __init__(self):
        self.callbacks = {}
        self.next_id = 0
        self.owner = threading.get_ident()

    def schedule(self, _delay, callback):
        if threading.get_ident() != self.owner:
            raise AssertionError("worker touched UI scheduler")
        self.next_id += 1
        self.callbacks[self.next_id] = callback
        return self.next_id

    def cancel(self, identifier):
        self.callbacks.pop(identifier, None)

    def pump(self):
        callbacks, self.callbacks = self.callbacks, {}
        for callback in callbacks.values():
            callback()


class BackgroundTaskTests(unittest.TestCase):
    def test_external_refresh_waits_for_delivery_then_retries_pending_domains(self):
        from services.workbench_ui.trading_account_panel_runtime import TradingAccountRuntimeMixin
        from services.workbench_ui.state_sync import STATE_ACCOUNT

        scheduler = FakeScheduler()
        runner = BackgroundTaskRunner(scheduler.schedule, scheduler.cancel)
        self.addCleanup(runner.close)
        panel = SimpleNamespace(
            _initial_state_thread=None, _initial_state_started=False,
            _initial_state_applied=True, _command_runner=runner,
            _external_state_refresh_pending_domains=set(),
            after_idle=lambda callback: scheduler.schedule(0, callback),
        )
        submitted = []
        def submit(_label, worker, **_kwargs):
            thread = runner.submit(worker, lambda _result: None)
            if thread is not None:
                submitted.append(thread)
            return thread is not None
        panel._submit_trading_command = submit
        panel.refresh_for_state_domains = lambda domains: TradingAccountRuntimeMixin.refresh_for_state_domains(panel, domains)
        worker = runner.submit(lambda: 1, lambda _result:
            TradingAccountRuntimeMixin._schedule_pending_external_account_refresh(panel))
        panel._command_thread = worker
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertTrue(runner.busy)
        self.assertFalse(panel.refresh_for_state_domains({STATE_ACCOUNT}))
        self.assertEqual({STATE_ACCOUNT}, panel._external_state_refresh_pending_domains)
        scheduler.pump()  # Deliver old result; completion schedules pending refresh.
        scheduler.pump()  # Submit the refresh only after delivery releases the slot.
        self.assertEqual(1, len(submitted))
        self.assertFalse(panel._external_state_refresh_pending_domains)
        submitted[0].join(5)
        scheduler.pump()
        panel._submit_trading_command = lambda *_args, **_kwargs: False
        self.assertFalse(panel.refresh_for_state_domains({STATE_ACCOUNT}))
        self.assertEqual({STATE_ACCOUNT}, panel._external_state_refresh_pending_domains)

    def test_replacement_error_and_close_discard_stale_results(self):
        scheduler = FakeScheduler()
        runner = BackgroundTaskRunner(scheduler.schedule, scheduler.cancel)
        release = threading.Event()
        results = []
        first = runner.submit(lambda: (release.wait(5), "old")[1], results.append)
        latest = runner.submit(lambda: "new", results.append, replace=True)
        latest.join(5)
        scheduler.pump()
        self.assertEqual(["new"], [row.value for row in results])
        release.set()
        first.join(5)
        def fail():
            raise ValueError("market evidence unavailable")
        failed = runner.submit(fail, results.append)
        failed.join(5)
        scheduler.pump()
        self.assertEqual(2, len(results))
        self.assertIsInstance(results[-1].error, ValueError)
        closed = runner.submit(lambda: "closed", results.append)
        runner.close()
        closed.join(5)
        scheduler.pump()
        self.assertEqual(2, len(results))
        self.assertFalse(scheduler.callbacks)

    def test_busy_includes_completed_but_undelivered_result(self):
        scheduler = FakeScheduler()
        runner = BackgroundTaskRunner(scheduler.schedule, scheduler.cancel)
        worker = runner.submit(lambda: 1, lambda _result: None)
        worker.join(5)
        self.assertTrue(runner.busy)
        self.assertIsNone(runner.submit(lambda: 2, lambda _result: None))
        scheduler.pump()
        self.assertFalse(runner.busy)


if __name__ == "__main__":
    unittest.main()
