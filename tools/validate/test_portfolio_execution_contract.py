"""Explicit Portfolio execution across nested scopes, caches and worker boundaries."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
from multiprocessing import get_context
import unittest

from core.portfolio_entry_policy import (
    PortfolioExecutionContract, DAILY_EXECUTION_LEGACY, DAILY_EXECUTION_FULL_FILTERS,
    DAILY_EXECUTION_ATR_LIMIT, portfolio_entry_policy_context, daily_signal_execution_context,
    current_daily_entry_source, current_daily_execution_contract, resolve_portfolio_execution_contract,
)


def prepare_explicit_worker(contract):
    from tools.validate.synthetic_daily_trade40_cases import fixture
    from services.optimizer.trial_inputs import worker_prep_data
    _, frame, params, _ = fixture(count=160)
    result = worker_prep_data(
        "2330", frame, replace(params, use_breakout_buy=False),
        include_trade_logs=False, include_pit_stats_index=False,
        profile_enabled=False, execution_contract=contract,
    )
    if not result["ok"]:
        raise AssertionError(result["reason"])
    return {key: sha256(value.tobytes()).hexdigest()
            for key, value in result["dynamic"].items()}


class PortfolioExecutionContractTests(unittest.TestCase):
    def test_nested_breakout_masks_daily_and_exception_restores_both(self):
        self.assertIsNone(current_daily_entry_source())
        with portfolio_entry_policy_context("Daily", unscored=True,
                daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS) as full:
            source = current_daily_entry_source()
            with self.assertRaisesRegex(RuntimeError, "probe"):
                with portfolio_entry_policy_context("Breakout") as breakout:
                    self.assertIsNone(current_daily_entry_source())
                    self.assertEqual(breakout, PortfolioExecutionContract())
                    self.assertEqual(current_daily_execution_contract(), DAILY_EXECUTION_LEGACY)
                    raise RuntimeError("probe")
            self.assertIs(current_daily_entry_source(), source)
            self.assertEqual(resolve_portfolio_execution_contract(), full)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                with daily_signal_execution_context(DAILY_EXECUTION_ATR_LIMIT):
                    self.fail("Inconsistent source/mode entered")
        self.assertIsNone(current_daily_entry_source())
        self.assertEqual(resolve_portfolio_execution_contract(), PortfolioExecutionContract())

    def test_contract_is_immutable_and_conflicting_inputs_fail(self):
        contract = PortfolioExecutionContract()
        self.assertEqual(PortfolioExecutionContract(daily_execution_contract=None), contract)
        with self.assertRaises(FrozenInstanceError):
            contract.strategy = "Daily"
        with self.assertRaises(ValueError):
            PortfolioExecutionContract("Breakout", DAILY_EXECUTION_FULL_FILTERS)
        with self.assertRaises(ValueError):
            resolve_portfolio_execution_contract(contract, daily_execution_contract=DAILY_EXECUTION_ATR_LIMIT)

    def test_cache_uses_supplied_contract_not_current_scope(self):
        from services.portfolio_replay import _replay_preparation_key
        record = {"params_signature": "signed", "effective_date_text": "2021-01-01"}
        kwargs = dict(ensemble=False, raw_universe_required_min_rows=None,
                      keep_trade_logs=False, use_prepared_cache=False, write_prepared_cache=False)
        breakout = PortfolioExecutionContract()
        baseline = _replay_preparation_key(".", [record], execution_contract=breakout, **kwargs)
        with portfolio_entry_policy_context("Daily", unscored=True,
                daily_execution_contract=DAILY_EXECUTION_FULL_FILTERS) as full:
            self.assertEqual(baseline, _replay_preparation_key(".", [record], execution_contract=breakout, **kwargs))
            self.assertNotEqual(baseline, _replay_preparation_key(".", [record], execution_contract=full, **kwargs))

    def test_inline_thread_spawn_prepare_identical_contracts(self):
        contracts = (PortfolioExecutionContract(), PortfolioExecutionContract(
            "Daily", DAILY_EXECUTION_FULL_FILTERS, "daily_unscored"))
        expected = list(map(prepare_explicit_worker, contracts))
        self.assertNotEqual(expected[0]["is_setup"], expected[1]["is_setup"])
        with portfolio_entry_policy_context("Daily", unscored=True,
                daily_execution_contract=DAILY_EXECUTION_ATR_LIMIT):
            with ThreadPoolExecutor(max_workers=2) as executor:
                self.assertEqual(expected, list(executor.map(prepare_explicit_worker, contracts)))
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as executor:
                self.assertEqual(expected, list(executor.map(prepare_explicit_worker, contracts)))

    def test_explicit_engine_contract_matches_scoped_entry(self):
        from tools.validate.synthetic_daily_trade40_oracle_cases import source_fixture
        from core.portfolio_fast_data import prep_stock_data_and_trades, pack_prepared_stock_data
        from core.portfolio_engine import run_portfolio_timeline
        from core.file_integrity import canonical_json_sha256
        from core.serialization_utils import json_native_value
        exchange, frames, _, schedule = source_fixture()
        params = schedule[0].params
        market, logs = {}, {}
        for ticker, frame in frames.items():
            ready, logs[ticker] = prep_stock_data_and_trades(frame, params)
            market[ticker] = pack_prepared_stock_data(ready)
        def run(contract=None):
            return run_portfolio_timeline(
                market, logs, list(exchange), 2021, params, 10, False,
                benchmark_ticker="2330", benchmark_data=market["2330"],
                is_training=False, verbose=False, execution_contract=contract,
            )
        with portfolio_entry_policy_context("Daily", unscored=True) as daily:
            scoped = run()
            breakout = run(PortfolioExecutionContract())
        explicit = run(daily)
        # Full tuple, including cash/trade ledger and terminal account statistics.
        self.assertEqual(canonical_json_sha256(json_native_value(scoped)),
                         canonical_json_sha256(json_native_value(explicit)))
        self.assertNotEqual(int(breakout[4]), int(explicit[4]))


if __name__ == "__main__":
    unittest.main()
