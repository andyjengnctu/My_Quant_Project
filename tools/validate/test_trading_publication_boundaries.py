"""Isolated public-writer, consumer CAS and Accounting delivery contracts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from core.file_integrity import atomic_write_json, canonical_json_sha256, load_json_strict
from core.process_lock import try_process_lock
from core.strategy_params import V16StrategyParams
from core.trading_fill_transaction import build_trading_fill_transaction_journal
from core.trading_order_state import build_empty_trading_order_state
from core.trading_state_paths import (
    resolve_trading_account_state_path, resolve_trading_order_state_path,
    resolve_trading_fill_transaction_path, resolve_trading_pending_entry_transaction_path,
)
from services.trading import account_state as accounts, order_state as orders, pending_entry_state as pending
from services.trading import market_data_consumer as consumers, market_data_consumer_promotion as promotion
from services.trading.fill_reconciliation import recover_trading_fill_transaction
from services.trading.state_lock import (
    TradingTransactionPendingError, TradingStateBusyError,
    assert_trading_state_writable, pending_entry_transaction_access, serialized_trading_state_mutation,
)
from services.workbench_ui import accounting_center_panel as accounting
from services.workbench_ui.background_tasks import BackgroundTaskRunner
from .test_trading_snapshot_tasks import FakeScheduler


class JournalGateTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.account = accounts.initialize_trading_account_state(self.root, cash=1000)
        self.orders = build_empty_trading_order_state(timestamp="2026-11-24T08:00:00+08:00", mutation_id="initial")
        atomic_write_json(resolve_trading_order_state_path(self.root), self.orders)

    def fill_journal(self):
        return build_trading_fill_transaction_journal(
            transaction_id="interrupted", created_at="2026-11-24T08:00:00+08:00",
            account_original=self.account, account_target=self.account,
            order_original=self.orders, order_target=self.orders,
        )

    def test_every_public_account_mutator_rejects_both_pending_journals_without_writes(self):
        # Discover canonical account command adapters so a future adapter cannot
        # silently lose this invariant. Arguments deliberately cannot mutate state.
        arguments = dict(ticker="2330", qty=1, price=10, trade_date="2026-11-23",
            cash=2000, expected_revision=None, cost_basis_total=10, entry_date="2026-11-23",
            management_lineage={}, position_state={}, management_start_date="2026-11-23",
            params=V16StrategyParams(), execution_plan_seed={}, strategy_lineage={},
            transaction_revision=1, updates={}, signals={})
        commands = [(name, fn) for name, fn in vars(accounts).items()
                    if not name.startswith("_") and inspect.isfunction(fn)
                    and fn.__module__ == accounts.__name__ and "_mutate_account(" in inspect.getsource(fn)]
        self.assertGreater(len(commands), 10)
        account_path = resolve_trading_account_state_path(self.root)
        before = account_path.read_bytes()
        for resolver in (resolve_trading_fill_transaction_path, resolve_trading_pending_entry_transaction_path):
            path = resolver(self.root)
            atomic_write_json(path, self.fill_journal() if resolver is resolve_trading_fill_transaction_path else {})
            for name, fn in commands:
                with self.subTest(journal=path.name, command=name):
                    kwargs = {key: value for key, value in arguments.items() if key in inspect.signature(fn).parameters}
                    with self.assertRaises(TradingTransactionPendingError):
                        fn(self.root, **kwargs)
                    self.assertEqual(before, account_path.read_bytes())
            path.unlink()

    def test_ui_failure_preserves_journal_recoverability_and_none_revision_policy(self):
        tx = resolve_trading_fill_transaction_path(self.root)
        atomic_write_json(tx, self.fill_journal())
        panel = SimpleNamespace(_parse_cash=lambda: 2000,
                                _refresh_after_account_mutation=lambda: self.fail("failed command refreshed"))
        with patch.object(accounting, "WORKBENCH_PROJECT_ROOT", self.root), \
             patch.object(accounting.messagebox, "showerror") as error:
            accounting.AccountingCenterPanel._set_cash(panel)
        error.assert_called_once()
        self.assertEqual(self.account, load_json_strict(resolve_trading_account_state_path(self.root)))
        self.assertTrue(recover_trading_fill_transaction(self.root))
        # Broker-truth corrections still use current revision and are independent
        # of the optional active-order policy once transaction recovery is complete.
        with patch.object(accounts, "_read_order_guard_sha", side_effect=AssertionError("active-order gate")):
            result = accounts.set_trading_cash_balance(self.root, cash=2000, expected_revision=None)
        self.assertEqual(2000000, result["cash_milli"])

    def test_order_pending_and_initializer_writers_reject_journals_before_validation(self):
        actions = [
            lambda: orders.mutate_trading_order_state(self.root, expected_revision=0, mutator=lambda *a: self.fail("mutated")),
            lambda: orders.confirm_trading_order_cancellation(self.root, order_id="missing", expected_revision=0),
            lambda: pending.create_trading_pending_entry(self.root, entry={}),
            lambda: pending.update_trading_pending_entry(self.root, pending_entry_id="missing", replacement={}),
            lambda: pending.cancel_trading_pending_entry(self.root, pending_entry_id="missing"),
            lambda: pending.delete_trading_pending_entry(self.root, pending_entry_id="missing"),
            lambda: pending.mark_trading_pending_entry_filled(self.root, pending_entry_id="missing", fill={}),
            lambda: accounts.initialize_trading_account_state(self.root, cash=0),
        ]
        for resolver in (resolve_trading_fill_transaction_path, resolve_trading_pending_entry_transaction_path):
            path = resolver(self.root)
            atomic_write_json(path, {})
            for action in actions:
                with self.assertRaises(TradingTransactionPendingError):
                    action()
            path.unlink()

    def test_pending_transfer_authority_requires_owned_lock_and_exact_journal(self):
        path = resolve_trading_pending_entry_transaction_path(self.root)
        atomic_write_json(path, {"transfer": "original"})
        with self.assertRaises(TradingStateBusyError):
            with pending_entry_transaction_access(self.root):
                pass
        @serialized_trading_state_mutation
        def transfer(root):
            with pending_entry_transaction_access(root):
                accounts.set_trading_cash_balance(root, cash=2000, expected_revision=None)
                atomic_write_json(path, {"transfer": "changed"})
                with self.assertRaises(TradingTransactionPendingError):
                    assert_trading_state_writable(root)
            with self.assertRaises(TradingTransactionPendingError):
                accounts.set_trading_cash_balance(root, cash=3000, expected_revision=None)
        transfer(self.root)
        self.assertEqual(2000000, load_json_strict(resolve_trading_account_state_path(self.root))["cash_milli"])

    def test_journal_appearing_during_mutation_prevents_commit(self):
        path = resolve_trading_account_state_path(self.root)
        before = path.read_bytes()
        original = accounts.set_trading_account_cash
        def prepare(*args, **kwargs):
            result = original(*args, **kwargs)
            atomic_write_json(resolve_trading_fill_transaction_path(self.root), self.fill_journal())
            return result
        with patch.object(accounts, "set_trading_account_cash", prepare), self.assertRaises(TradingTransactionPendingError):
            accounts.set_trading_cash_balance(self.root, cash=2000, expected_revision=None)
        self.assertEqual(before, path.read_bytes())
        self.assertTrue(recover_trading_fill_transaction(self.root))

    def test_pending_transfer_failure_keeps_recovery_authority_and_blocks_unrelated_writes(self):
        from services.trading import pending_entry_service as transfers
        from services.trading.strategy_param_runtime import build_trading_manual_management_lineage
        params = V16StrategyParams()
        seed = dict(entry_type="manual", qty=1, limit_price=100.0, init_sl=90.0,
                    init_trail=92.0, target_price=120.0, entry_atr=5.0,
                    reserved_cost=100.0, reserved_cost_milli=100000,
                    security_profile={}, trade_date="2026-11-20")
        lineage = build_trading_manual_management_lineage(
            params=params, execution_plan_seed=seed, information_date="2026-11-20",
            origin="manual_pending_entry", planned_qty=1, planned_cost=100.0,
        )
        entry = pending.create_trading_pending_entry(self.root, entry=dict(
            origin="manual_selected", ticker="2330", information_date="2026-11-20",
            planned_trade_date="2026-11-20", execution_plan_seed=seed, planned_qty=1,
            reserved_cost_milli=100000, management_lineage=lineage,
        ))
        tx = resolve_trading_pending_entry_transaction_path(self.root)
        with ExitStack() as stack:
            stack.enter_context(patch.object(transfers, "preview_trading_pending_entry_fill",
                                             return_value={"entry": entry}))
            stack.enter_context(patch.object(transfers, "resolve_trading_lifecycle_context",
                                             return_value=SimpleNamespace(finalized_date="2026-11-24", view=object())))
            stack.enter_context(patch.object(transfers, "load_trading_v2_sanitized_ohlcv_frame", return_value=object()))
            stack.enter_context(patch.object(transfers, "resolve_confirmed_entry_plan_from_frame", return_value=seed))
            def transfer():
                return transfers.fill_trading_pending_entry(
                    self.root, pending_entry_id=entry["pending_entry_id"], qty=1,
                    price=100.0, trade_date="2026-11-23", expected_account_revision=None,
                )
            with patch.object(transfers, "record_managed_manual_trading_buy", side_effect=OSError("before commit")):
                with self.assertRaisesRegex(OSError, "before commit"):
                    transfer()
            self.assertFalse(tx.exists())
            self.assertEqual(self.account, accounts.load_trading_account_state(self.root))
            with patch.object(transfers, "mark_trading_pending_entry_filled", side_effect=OSError("after account commit")):
                with self.assertRaisesRegex(OSError, "after account commit"):
                    transfer()
        self.assertTrue(tx.exists())
        committed = accounts.load_trading_account_state(self.root)
        self.assertEqual(1, committed["positions"]["2330"]["broker"]["initial_qty"])
        with self.assertRaises(TradingTransactionPendingError):
            accounts.set_trading_cash_balance(self.root, cash=2000, expected_revision=None)
        recovery = transfers.recover_trading_pending_entry_transaction(self.root)
        self.assertEqual("RECOVERED_FILLED", recovery["status"])
        self.assertFalse(tx.exists())
        self.assertEqual(committed, accounts.load_trading_account_state(self.root))
        self.assertEqual("FILLED", pending.load_trading_pending_entry_state(self.root)["entries"][entry["pending_entry_id"]]["status"])
        self.assertIsNone(transfers.recover_trading_pending_entry_transaction(self.root))


class ConsumerView:
    def __init__(self, batches=(), barrier=None, provider_date="2026-11-20"):
        self.archive = SimpleNamespace(snapshot_fingerprint="a" * 64, as_of_date=provider_date)
        self.batches = tuple(batches)
        self.barrier = barrier

    def training_horizon(self, **kwargs):
        return SimpleNamespace(training_through_date=kwargs["maximum_training_date"])

    def view_identity(self, **kwargs):
        if self.barrier:
            entered, release = self.barrier
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test publication barrier")
        return {"view_fingerprint": canonical_json_sha256([self.batches, kwargs]),
                "overlay_batch_fingerprints": list(self.batches)}

    def overlay_batch_fingerprints_for_datasets(self, datasets):
        return self.batches


class ConsumerPublicationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stack = self.enterContext(ExitStack())
        self.stack.enter_context(patch.object(consumers, "_required_v2_datasets", return_value=("prices",)))
        self.stack.enter_context(patch.object(consumers, "resolve_trading_v2_current_execution_pool", return_value=(["2330"], {})))

    def publish(self, date, expected, view=None):
        return consumers.publish_trading_v2_consumer_state(self.root, market_date=date,
            expected_state_fingerprint=expected, view=view or ConsumerView())

    def test_cas_prevents_interleaved_same_day_revision_and_old_date(self):
        original = self.publish("2026-11-23", None, ConsumerView(("base",)))
        for date in ("2026-11-23", "2026-11-24"):
            entered, release = Event(), Event()
            with ThreadPoolExecutor(max_workers=1) as pool:
                stale = pool.submit(self.publish, date, original["state_fingerprint"],
                                    ConsumerView(("base", "first"), (entered, release)))
                self.assertTrue(entered.wait(5))
                try:
                    newer = self.publish(date, original["state_fingerprint"], ConsumerView(("base", "first", "latest")))
                finally:
                    release.set()
                with self.assertRaises(consumers.TradingConsumerPublicationConflict):
                    stale.result(timeout=5)
            self.assertEqual(newer, consumers.load_trading_v2_consumer_state(self.root))
            original = newer
        with self.assertRaises(consumers.TradingConsumerPublicationConflict):
            self.publish("2026-11-23", original["state_fingerprint"])

    def test_same_day_cannot_discard_batches_even_with_current_expected_fingerprint(self):
        prior = self.publish("2026-11-23", None, ConsumerView(("base", "revision")))
        with self.assertRaises(consumers.TradingConsumerPublicationConflict):
            self.publish("2026-11-23", prior["state_fingerprint"], ConsumerView(("base",)))
        self.assertEqual(prior, consumers.load_trading_v2_consumer_state(self.root))

    def test_conflict_replans_formal_promotion_instead_of_reusing_old_view(self):
        entered, release = Event(), Event()
        for name, value in (("find_latest_ready_provider_snapshot", object()),
                            ("get_trading_strategy_profile", SimpleNamespace(strategy_id="test")),
                            ("load_market_data_dataset_state", {}),
                            ("build_trading_data_readiness_from_evidence", {"ready": True}),
                            ("load_trading_account_state", {}),
                            ("resolve_trading_live_reentry_required_tickers", []),
                            ("resolve_trading_market_data_update_target_date", "2026-11-23")):
            self.stack.enter_context(patch.object(promotion, name, return_value=value))
        self.stack.enter_context(patch.object(consumers.TradingMarketDataV2View, "open_as_of_target",
            side_effect=lambda _root, target_date: ConsumerView(barrier=(entered, release) if target_date == "2026-11-23" else None)))
        with ThreadPoolExecutor(max_workers=1) as pool:
            old = pool.submit(promotion.reconcile_trading_v2_consumer_state_from_local_evidence, self.root)
            self.assertTrue(entered.wait(5))
            try:
                latest = promotion.promote_trading_v2_consumer_state_if_ready(self.root, target_date="2026-11-24")
            finally:
                release.set()
            self.assertTrue(latest["promoted"])
            self.assertEqual("ALREADY_FINALIZED", old.result(timeout=5)["reason"])
        self.assertEqual("2026-11-24", consumers.load_trading_v2_consumer_state(self.root)["market_date"])

    def test_process_lock_and_write_failure_leave_current_state_unchanged(self):
        prior = self.publish("2026-11-23", None)
        path = self.root / consumers.TRADING_V2_CONSUMER_STATE_RELATIVE_PATH
        before = path.read_bytes()
        with try_process_lock(path.with_suffix(".publication.sqlite3")) as acquired:
            self.assertTrue(acquired)
            with self.assertRaises(consumers.TradingConsumerPublicationConflict):
                self.publish("2026-11-24", prior["state_fingerprint"])
        with patch.object(consumers, "atomic_write_json", side_effect=OSError("injected disk failure")):
            with self.assertRaises(OSError):
                self.publish("2026-11-24", prior["state_fingerprint"])
        self.assertEqual(before, path.read_bytes())
        self.assertEqual("2026-11-24", self.publish("2026-11-24", prior["state_fingerprint"])["market_date"])


class HeadlessAccounting(accounting.AccountingCenterPanel):
    def __init__(self):
        self.scheduler = FakeScheduler()
        self.after = self.scheduler.schedule
        self.after_cancel = self.scheduler.cancel
        self.after_idle = lambda callback: self.after(0, callback)
        self._refresh_runner = BackgroundTaskRunner(self.after, self.after_cancel)
        self._account_command_runner = BackgroundTaskRunner(self.after, self.after_cancel)
        self._refresh_generation = 0
        self._refresh_pending = False
        self._workbench_active = True
        self._deferred_refresh_result = None
        self._startup_refresh_started = True
        self._startup_refresh_after_id = self._stock_name_warm_after_id = None
        self._snapshot = self._dashboard = {}
        self.displayed = []
        self.status = []
        self._refresh_status_var = SimpleNamespace(set=self.status.append)
        self._status_var = self._market_date_var = SimpleNamespace(set=lambda _value: None)
        self._init_button = self._cash_button = SimpleNamespace(configure=lambda **_kwargs: None)

    def _ensure_stock_name_warm(self):
        pass

    def _render(self):
        self.displayed.append(self._snapshot["revision"])


class AccountingRefreshTests(unittest.TestCase):
    def test_completed_undelivered_refresh_coalesces_and_delivers_next_result(self):
        panel = HeadlessAccounting()
        self.addCleanup(panel._refresh_runner.close)
        with patch.object(accounting, "build_trading_accounting_snapshot", side_effect=[({"revision": 1}, {}), ({"revision": 2}, {})]) as load:
            panel.refresh()
            panel._refresh_runner._thread.join(5)
            panel.refresh()
            self.assertTrue(panel._refresh_pending)
            self.assertEqual(1, load.call_count)
            panel.scheduler.pump()
            panel.scheduler.pump()
            panel._refresh_runner._thread.join(5)
            panel.scheduler.pump()
        self.assertEqual([1, 2], panel.displayed)
        self.assertFalse(panel._refresh_runner.busy)

    def test_error_hidden_delivery_and_destroy_are_generation_safe(self):
        panel = HeadlessAccounting()
        self.addCleanup(panel._refresh_runner.close)
        with patch.object(accounting, "build_trading_accounting_snapshot", side_effect=[ValueError("source unavailable"), ({"revision": 2}, {}), ({"revision": 3}, {})]):
            panel.refresh()
            panel._refresh_runner._thread.join(5)
            panel.scheduler.pump()
            self.assertIn("source unavailable", panel.status[-1])
            panel._workbench_active = False
            panel.refresh()
            panel._refresh_runner._thread.join(5)
            panel.scheduler.pump()
            self.assertFalse(panel.displayed)
            panel.on_workbench_visibility_changed(True)  # queues old idle render
            panel.refresh()  # a newer generation makes the queued render obsolete
            panel._refresh_runner._thread.join(5)
            panel.scheduler.pump()
            self.assertEqual([3], panel.displayed)
        entered, release = Event(), Event()
        def blocked(_root):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test accounting close barrier")
            return {"revision": 4}, {}
        with patch.object(accounting, "build_trading_accounting_snapshot", blocked):
            panel.refresh()
            worker = panel._refresh_runner._thread
            self.assertTrue(entered.wait(5))
            with patch.object(accounting.ttk.Frame, "destroy"):
                panel.destroy()
            release.set()
            worker.join(5)
            panel.scheduler.pump()
        self.assertEqual([3], panel.displayed)


if __name__ == "__main__":
    unittest.main()
