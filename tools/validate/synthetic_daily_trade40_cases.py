"""Canonical Min ROOS signal-aligned 40/60/full-cycle regressions.

The retired MR-13CL P2/P16 teacher is deliberately not an executable path.
No formal Windows suite, fitting, optimizer, or real-market Replay is run here.
"""
from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.strategy_params import V16StrategyParams
from core.portfolio_param_runtime import build_portfolio_params_signature
from config.breakout_quality import BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE
from filters.breakout_quality.daily_trade40_target import (
    HORIZON_SESSIONS, ExecutableMinPeriod, build_formal_min_execution_schedule,
    compute_counterfactual_trade40_batch, resolve_historical_min_period,
    collect_canonical_exchange_sessions,
    resolve_formal_min_param_records,
)
from filters.breakout_quality.trade_path_label import (
    build_signal_cache, simulate_realized_trade_path_label,
)
from core.price_utils import adjust_long_buy_limit_array
from filters.breakout_quality.daily_ranker_data import load_daily_universal_ranker_data
from core.breakout_quality_registry import get_continuous_ranker_research_spec

OLD_PROFILE = "daily_universal_counterfactual_mintrade40_netr_full_list_ndcg_pairwise"


def fixture(*, count=130, rising=True, start="2023-01-02"):
    days = pd.bdate_range(start, periods=count)
    price = 100.0 + (0.22 if rising else -0.15) * np.arange(count)
    frame = pd.DataFrame({
        "Open": price, "High": price + 2, "Low": price - 2,
        "Close": price, "Volume": np.full(count, 1_000_000.),
    }, index=days)
    # Genuine canonical Min buy setups are upward crossovers of prior highs,
    # not arbitrary daily Close+ATR hypotheticals. Use actual OHLCV to produce
    # the known test-origin signals at 48/75/90, without mocking the producer.
    frame["Open"] = frame["Close"] - 0.25
    if rising:
        for index in (48, 75, 90):
            if index < count:
                for column in ("Open", "High", "Low", "Close"):
                    frame.iloc[index - 1, frame.columns.get_loc(column)] -= 7.0
                    frame.iloc[index:, frame.columns.get_loc(column)] += 8.0
    frame.attrs["ticker"] = "2330"
    params = V16StrategyParams(
        high_len=10, atr_len=5, use_breakout_ema_filter=False,
        use_bb=False, use_kc=False, use_vol=False,
        use_breakout_return_filter=False, use_breakout_false_filter=False,
        use_history_threshold=False, use_breakout_quality_filter=False,
        use_breakout_quality_ranking=False, atr_buy_tol=1.5,
    )
    signature = build_portfolio_params_signature(params)
    period = ExecutableMinPeriod(days[0], None, params, "formal_min_roos_seed", signature)
    return days, frame, params, period


class FormalMinTrade40SyntheticCases(unittest.TestCase):
    def test_net_r_daily_truth_uses_canonical_dl_float32_rank(self):
        from filters.breakout_quality.daily_trade40_target import rank_trade40_outcome_frame
        from filters.breakout_quality.continuous_ranker_data import build_same_date_percentile_targets
        sample = pd.DataFrame({
            "ticker": ["2330", "2317", "2303", "2454", "2408"],
            "date": ["2023-01-03"] * 5,
            "realized_net_r": [2., 1., 1., None, 0.],
        })
        result = rank_trade40_outcome_frame(sample)
        sentinel = -1.0
        label = build_same_date_percentile_targets(
            np.array([2., 1., 1., sentinel, 0.]),
            np.ones(len(sample), dtype=bool), sample["date"],
        )
        for index, row in sample.iterrows():
            self.assertEqual(result[(row["ticker"], row["date"])], float(label[index]))
        self.assertTrue(pd.isna(sample.loc[3, "realized_net_r"]))

    def test_mr13cl_is_retired_and_historical_id_is_immutable(self):
        self.assertNotEqual(BREAKOUT_QUALITY_MODEL_RESEARCH_MODEL_PROFILE[0], "MR-13CL")
        spec = get_continuous_ranker_research_spec(OLD_PROFILE)
        self.assertEqual(spec.model_research_id, "MR-13CL")
        self.assertFalse(spec.selection_pit_authorized)
        self.assertFalse(spec.current_time_validation_authorized)
        with patch("filters.breakout_quality.daily_ranker_data.load_validated_dataset_bundle") as source:
            with self.assertRaisesRegex(ValueError, "MR-13CL target retired"):
                load_daily_universal_ranker_data(
                    filter_id="breakout_quality_v1", model_architecture="inception_time_v1",
                    experiment_profile=OLD_PROFILE, preload_feature_bank=False,
                    allow_stale_source=True, project_root=Path("/missing"),
                )
            source.assert_not_called()

    def test_60_exchange_sessions_change_only_maturity_and_forced_exit(self):
        from unittest.mock import patch
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        schedule = (period,)
        with patch.object(owner, "simulate_realized_trade_path_label",
                          wraps=simulate_realized_trade_path_label) as ledger:
            forty = compute_counterfactual_trade40_batch(
                frame, np.array([75]), ticker="2330", schedule=schedule,
                exchange_calendar=days,
            )
            sixty = compute_counterfactual_trade40_batch(
                frame, np.array([75]), ticker="2330", schedule=schedule,
                exchange_calendar=days, horizon_sessions=60,
            )
        self.assertTrue(forty.valid_mask[0])
        self.assertTrue(sixty.valid_mask[0])
        self.assertEqual(forty.maturity_dates[0],
                         days[115].to_datetime64().astype("datetime64[D]"))
        self.assertEqual(sixty.maturity_dates[0],
                         days[135].to_datetime64().astype("datetime64[D]"))
        self.assertEqual([call.kwargs["horizon_end_date"] for call in ledger.call_args_list],
                         [days[115], days[135]])
        self.assertEqual([call.kwargs["params"] for call in ledger.call_args_list],
                         [period.params, period.params])
        self.assertEqual(ledger.call_count, 2)
        # The same stock-day remains eligible for 60D. When the dataset
        # ends first, it is explicitly closed out at Research-data cutoff.
        forty_late = compute_counterfactual_trade40_batch(
            frame, np.array([90]), ticker="2330", schedule=schedule,
            exchange_calendar=days,
        )
        sixty_late = compute_counterfactual_trade40_batch(
            frame, np.array([90]), ticker="2330", schedule=schedule,
            exchange_calendar=days, horizon_sessions=60,
        )
        self.assertTrue(sixty_late.valid_mask[0])
        self.assertTrue(forty_late.valid_mask[0])
        self.assertEqual(sixty_late.maturity_dates[0], days[-1].to_datetime64().astype("datetime64[D]"))
        self.assertEqual(sixty_late.cutoff_truncated_rows, 1)
        self.assertEqual(sixty_late.research_cutoff_closeout_rows, 1)
        with self.assertRaisesRegex(ValueError, "positive exchange sessions"):
            compute_counterfactual_trade40_batch(
                frame, np.array([10]), ticker="2330", schedule=schedule,
                exchange_calendar=days, horizon_sessions=0,
            )

    def test_full_cycle_netpnl_is_the_canonical_ledger_amount_not_a_second_formula(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        recorded = []
        canonical_path = simulate_realized_trade_path_label
        def trace(*args, **kwargs):
            observed = canonical_path(*args, **kwargs)
            recorded.append(observed)
            return observed
        inputs = dict(frame=frame, positions=np.asarray([75]), ticker="2330",
                      schedule=(period,), exchange_calendar=days,
                      horizon_sessions=None, research_cutoff=days[120])
        with patch.object(owner, "simulate_realized_trade_path_label", side_effect=trace):
            dual = compute_counterfactual_trade40_batch(**inputs, include_net_pnl=True)
        self.assertEqual(len(recorded), 1)
        self.assertTrue(dual.valid_mask[0])
        self.assertIsNotNone(dual.target_raw_pnl)
        self.assertAlmostEqual(dual.target_raw_pnl[0], recorded[0].realized_net_pnl, places=8)
        self.assertAlmostEqual(dual.target_raw_r[0], recorded[0].realized_net_r, places=5)
        self.assertEqual(recorded[0].forced_closeout, True)
        self.assertGreater(recorded[0].sizing_capital, 0)
        default = compute_counterfactual_trade40_batch(**inputs)
        self.assertIsNone(default.target_raw_pnl)
        np.testing.assert_array_equal(default.target_raw_r, dual.target_raw_r)
        np.testing.assert_array_equal(default.maturity_dates, dual.maturity_dates)
        cutoff_censored = compute_counterfactual_trade40_batch(
            frame, np.asarray([120]), ticker="2330", schedule=(period,),
            exchange_calendar=days, horizon_sessions=None,
            research_cutoff=days[120], include_net_pnl=True,
        )
        self.assertFalse(cutoff_censored.valid_mask[0])
        self.assertTrue(np.isnan(cutoff_censored.target_raw_pnl[0]))

    def test_netpnl_rejects_missing_or_disagreeing_canonical_ledger(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        actual = SimpleNamespace(
            realized_net_r=2.0, realized_net_pnl=None,
            reason="realized_net_profit", exit_date=str(days[100].date()),
            label_eval_end_date=str(days[100].date()), forced_closeout=False,
        )
        with patch.object(owner, "simulate_realized_trade_path_label", return_value=actual):
            with self.assertRaisesRegex(ValueError, "lacks finite broker-ledger NetPnL"):
                compute_counterfactual_trade40_batch(
                    frame, np.asarray([75]), ticker="2330", schedule=(period,),
                    exchange_calendar=days, horizon_sessions=None,
                    include_net_pnl=True,
                )
        actual.realized_net_pnl = -100.0
        with patch.object(owner, "simulate_realized_trade_path_label", return_value=actual):
            with self.assertRaisesRegex(ValueError, "signs disagree"):
                compute_counterfactual_trade40_batch(
                    frame, np.asarray([75]), ticker="2330", schedule=(period,),
                    exchange_calendar=days, horizon_sessions=None,
                    include_net_pnl=True,
                )

    def test_netpnl_rank_uses_amount_not_risk_normalized_net_r(self):
        sample = pd.DataFrame({
            "ticker": ["2330", "2317", "1101", "0050"],
            "date": ["2025-01-02"] * 4,
            "realized_net_r": [3.0, 1.5, -1.0, np.nan],
            "realized_net_pnl": [6000.0, 13500.0, -1500.0, np.nan],
        })
        from filters.breakout_quality.daily_trade40_target import rank_trade40_outcome_frame
        r = rank_trade40_outcome_frame(sample)
        pnl = rank_trade40_outcome_frame(sample, value_column="realized_net_pnl")
        self.assertGreater(r[("2330", "2025-01-02")], r[("2317", "2025-01-02")])
        self.assertLess(pnl[("2330", "2025-01-02")], pnl[("2317", "2025-01-02")])
        self.assertEqual(pnl[("0050", "2025-01-02")], 0.0)
        with self.assertRaisesRegex(ValueError, "not an approved economic metric"):
            rank_trade40_outcome_frame(sample, value_column="future_portfolio_equity")

    def test_any_horizon_and_full_cycle_use_one_earliest_settlement_rule(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        position = 75
        research_end = days[120]
        deadlines = []
        values = {}
        with patch.object(owner, "simulate_realized_trade_path_label",
                          wraps=simulate_realized_trade_path_label) as canonical_path:
            for horizon in (40, 60, 73, 100, None):
                batch = compute_counterfactual_trade40_batch(
                    frame, np.asarray([position]), ticker="2330", schedule=(period,),
                    exchange_calendar=days, horizon_sessions=horizon,
                    research_cutoff=research_end,
                )
                self.assertTrue(batch.valid_mask[0], f"horizon={horizon}")
                expected_end = days[115] if horizon == 40 else research_end
                self.assertEqual(batch.maturity_dates[0],
                                 expected_end.to_datetime64().astype("datetime64[D]"))
                self.assertEqual(batch.cutoff_truncated_rows, int(horizon != 40))
                self.assertEqual(batch.research_cutoff_closeout_rows, int(horizon != 40))
                values[horizon] = float(batch.target_raw_r[0])
                deadlines.append(canonical_path.call_args.kwargs["horizon_end_date"])
            self.assertEqual(canonical_path.call_count, 5)
        self.assertEqual(deadlines, [days[115], research_end, research_end,
                                     research_end, research_end])
        self.assertEqual(values[60], values[100])
        self.assertEqual(values[60], values[None])
        self.assertNotEqual(values[40], values[60])

        # An information bar on the last available Research session cannot
        # create a hypothetical next-day fill by borrowing post-cutoff data.
        barred = compute_counterfactual_trade40_batch(
            frame, np.asarray([120]), ticker="2330", schedule=(period,),
            exchange_calendar=days, horizon_sessions=None,
            research_cutoff=research_end,
        )
        self.assertFalse(barred.valid_mask[0])
        self.assertTrue(np.isnan(barred.target_raw_r[0]))
        self.assertEqual(barred.reasons[0], "no_future_executable_research_session")

    def test_label_maturity_follows_natural_exit_not_horizon_or_research_cutoff(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        signal = 75
        natural_exit = days[80]
        cutoff = days[120]
        actual = SimpleNamespace(
            realized_net_r=1.25, reason="realized_net_profit",
            exit_date=str(natural_exit.date()), label_eval_end_date=str(natural_exit.date()),
            forced_closeout=False,
        )
        with patch.object(owner, "simulate_realized_trade_path_label", return_value=actual):
            batch = compute_counterfactual_trade40_batch(
                frame, np.asarray([signal]), ticker="2330", schedule=(period,),
                exchange_calendar=days, horizon_sessions=None, research_cutoff=cutoff,
            )
        self.assertTrue(batch.valid_mask[0])
        self.assertEqual(batch.maturity_dates[0],
                         natural_exit.to_datetime64().astype("datetime64[D]"))
        # A horizon beyond the data cutoff is only potential truncation;
        # natural early exits are not actual Research-cutoff closeouts.
        self.assertEqual(batch.cutoff_truncated_rows, 1)
        self.assertEqual(batch.research_cutoff_closeout_rows, 0)

        censored_end = days[82]
        superseded = SimpleNamespace(
            realized_net_r=None, reason="unfilled_superseded_by_new_setup",
            exit_date=None, label_eval_end_date=str(censored_end.date()),
            forced_closeout=False,
        )
        with patch.object(owner, "simulate_realized_trade_path_label", return_value=superseded):
            censored = compute_counterfactual_trade40_batch(
                frame, np.asarray([signal]), ticker="2330", schedule=(period,),
                exchange_calendar=days, horizon_sessions=None, research_cutoff=cutoff,
            )
        self.assertFalse(censored.valid_mask[0])
        self.assertTrue(np.isnan(censored.target_raw_r[0]))
        self.assertEqual(censored.maturity_dates[0],
                         censored_end.to_datetime64().astype("datetime64[D]"))
        self.assertEqual(censored.research_cutoff_closeout_rows, 0)

        illegal = SimpleNamespace(
            realized_net_r=1.0, reason="realized_net_profit",
            exit_date=str(days[121].date()), label_eval_end_date=str(days[121].date()),
            forced_closeout=False,
        )
        with patch.object(owner, "simulate_realized_trade_path_label", return_value=illegal):
            with self.assertRaisesRegex(ValueError, "outside its legal observation window"):
                compute_counterfactual_trade40_batch(
                    frame, np.asarray([signal]), ticker="2330", schedule=(period,),
                    exchange_calendar=days, horizon_sessions=None, research_cutoff=cutoff,
                )

    def test_post_research_cutoff_quotes_cannot_affect_full_cycle_truth(self):
        days, frame, _params, period = fixture(count=150)
        cutoff = days[110]
        kwargs = dict(ticker="2330", schedule=(period,), exchange_calendar=days,
                      research_cutoff=cutoff, horizon_sessions=None)
        before = compute_counterfactual_trade40_batch(frame, np.asarray([70]), **kwargs)
        altered = frame.copy()
        after_end = altered.index > cutoff
        altered.loc[after_end, ["Open", "High", "Low", "Close"]] *= 7
        altered.loc[after_end, "Volume"] = 0
        after = compute_counterfactual_trade40_batch(altered, np.asarray([70]), **kwargs)
        self.assertEqual(before.valid_mask.tolist(), after.valid_mask.tolist())
        self.assertTrue(before.valid_mask[0])
        np.testing.assert_allclose(before.target_raw_r, after.target_raw_r, rtol=0, atol=0)
        np.testing.assert_array_equal(before.maturity_dates, after.maturity_dates)
        self.assertEqual(before.research_cutoff_closeout_rows, 1)

    def test_formal_seed_param_adapter_no_old_teacher_dependency(self):
        days, frame, params, period = fixture()
        signature = build_portfolio_params_signature(params)
        rows = [{"effective_date": date(2021, 1, 1), "effective_end_date": None,
                 "params_obj": params, "params_signature": signature}]
        formal, provenance = build_formal_min_execution_schedule(
            rows, seed=693545351, param_sha256="a" * 64,
        )
        self.assertEqual(formal[0].source_id, "formal_min_roos_seed")
        self.assertEqual(provenance["seed"], 693545351)
        self.assertEqual(provenance["param_identity_sha256"], "a" * 64)
        self.assertNotIn("P2", str(provenance))
        with self.assertRaisesRegex(ValueError, "canonical parameter identity"):
            build_formal_min_execution_schedule(rows, seed=1, param_sha256="invalid")

    def test_formal_schedule_requires_dl_off_and_nonoverlapping_periods(self):
        days, frame, params, period = fixture()
        sig = period.params_signature
        duplicate = [
            {"effective_date": "2021-01-01", "params_obj": params, "params_signature": sig},
            {"effective_date": "2021-01-01", "params_obj": params, "params_signature": sig},
        ]
        with self.assertRaisesRegex(ValueError, "unique ordered"):
            build_formal_min_execution_schedule(duplicate, seed=1, param_sha256="a" * 64)
        bad = V16StrategyParams(**{**params.__dict__, "use_breakout_quality_filter": True})
        with self.assertRaisesRegex(ValueError, "DL-off"):
            build_formal_min_execution_schedule([
                {"effective_date": "2021-01-01", "params_obj": bad, "params_signature": sig}],
                seed=1, param_sha256="a" * 64)

    def test_order_date_not_information_date_selects_effective_params(self):
        days, frame, params, _ = fixture(count=149)
        # Force 2020-12-31 info -> 2021-01-04 order, no 2021-01-01 holiday bar.
        trading_days = pd.bdate_range("2020-10-01", periods=150).difference(
            pd.DatetimeIndex(["2021-01-01"]))
        frame = frame.iloc[:len(trading_days)].copy()
        frame.index = trading_days
        idx = int(trading_days.get_loc(pd.Timestamp("2020-12-31")))
        frame.loc[frame.index[idx - 1], ["Open", "High", "Low", "Close"]] -= 7.0
        frame.loc[frame.index[idx]:, ["Open", "High", "Low", "Close"]] += 8.0
        frame["Open"] = frame["Close"] - 0.25
        formal = (ExecutableMinPeriod(pd.Timestamp("2021-01-01"), None,
                                     params, "formal_min_roos_seed", "sig"),)
        self.assertIsNone(resolve_historical_min_period(formal, trading_days[idx]))
        self.assertIs(resolve_historical_min_period(formal, trading_days[idx + 1]), formal[0])
        batch = compute_counterfactual_trade40_batch(
            frame, np.asarray([idx]), ticker="2330", schedule=formal,
            exchange_calendar=trading_days,
        )
        self.assertEqual(batch.reasons[0], "filled_executed_net_R")
        self.assertTrue(batch.valid_mask[0])
        self.assertEqual(trading_days[idx + 1], pd.Timestamp("2021-01-04"))
        # An explicitly different 2020 Min schedule must never control a
        # 2021-01-04 order merely because the information bar was 2020-12-31.
        from dataclasses import replace
        old_params = replace(params, atr_buy_tol=0.0)
        old_period = ExecutableMinPeriod(
            pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
            old_params, "formal_min_roos_seed",
            build_portfolio_params_signature(old_params),
        )
        paired = compute_counterfactual_trade40_batch(
            frame, np.asarray([idx]), ticker="2330",
            schedule=(old_period, formal[0]), exchange_calendar=trading_days,
        )
        self.assertEqual(paired.reasons, batch.reasons)
        np.testing.assert_allclose(paired.target_raw_r, batch.target_raw_r)

    def test_real_min_setup_and_hypothetical_override_use_same_ledger(self):
        params = V16StrategyParams(
            use_breakout_ema_filter=False, use_bb=False, use_kc=False,
            use_vol=False, use_breakout_return_filter=False,
            use_breakout_false_filter=False, use_history_threshold=False,
            use_breakout_reclaim_reentry=False, use_breakout_quality_filter=False,
            use_breakout_quality_ranking=False, high_len=3, atr_len=3,
            atr_buy_tol=1.0, atr_times_init=2.0, atr_times_trail=2.0,
            tp_percent=0.0, initial_capital=1_000_000.0,
            fixed_risk=0.01, max_position_cap_pct=0.3,
        )
        dates = pd.date_range("2020-01-01", periods=9, freq="B")
        frame = pd.DataFrame({
            "Open": [95, 98, 100, 103, 101, 103, 104, 105, 1100],
            "High": [97, 100, 102, 103.5, 102, 105, 105, 106, 1101],
            "Low": [94, 97, 99, 102, 100, 102, 103, 104, 1099],
            "Close": [96, 99, 101, 103, 101, 104, 104, 105, 1100],
            "Volume": [1000.0] * 9,
        }, index=dates)
        signals = (
            np.full(9, 2.0),
            np.array([False, False, True] + [False] * 6),
            np.zeros(9, dtype=bool),
            np.array([np.nan, np.nan, 101.0] + [np.nan] * 6),
        )
        original = simulate_realized_trade_path_label(
            frame, ticker="2330", signal_pos=2, params=params,
            teacher_effective_date="2020-01-01", precomputed_signals=signals,
            horizon_end_date=frame.index[5],
        )
        hypothetical = simulate_realized_trade_path_label(
            frame, ticker="2330", signal_pos=2, params=params,
            teacher_effective_date="2020-01-01", precomputed_signals=signals,
            horizon_end_date=frame.index[5], counterfactual_entry_limit=101.,
        )
        self.assertEqual(original.realized_net_r, hypothetical.realized_net_r)
        self.assertEqual(original.realized_net_pnl, hypothetical.realized_net_pnl)
        self.assertEqual(original.exit_date, hypothetical.exit_date)

    def test_hypothetical_target_forced_40th_and_stop_accounting(self):
        days, frame, params, period = fixture()
        signal_pos = 48
        oracle = compute_counterfactual_trade40_batch(
            frame, np.asarray([signal_pos]), ticker="2330",
            schedule=(period,), exchange_calendar=days,
        )
        self.assertTrue(oracle.valid_mask[0])
        self.assertEqual(oracle.maturity_dates[0],
                         days[signal_pos + HORIZON_SESSIONS].to_datetime64().astype("datetime64[D]"))
        signal_cache = build_signal_cache(frame, params, ticker="2330")
        atr = float(signal_cache[0][signal_pos])
        limit = float(adjust_long_buy_limit_array(np.asarray([
            float(frame["Close"].iloc[signal_pos]) + atr * float(params.atr_buy_tol)
        ]), ticker="2330")[0])
        expected = simulate_realized_trade_path_label(
            frame, ticker="2330", signal_pos=signal_pos, params=params,
            teacher_effective_date=str(period.start.date()),
            precomputed_signals=signal_cache, horizon_end_date=days[signal_pos + 40],
            counterfactual_entry_limit=limit,
        )
        self.assertTrue(expected.forced_closeout)
        self.assertEqual(expected.exit_reason, "HORIZON_CLOSEOUT")
        self.assertAlmostEqual(float(oracle.target_raw_r[0]), expected.realized_net_r, places=6)

    def test_missing_future_bar_does_not_fabricate_trades(self):
        days, frame, params, period = fixture()
        batch = compute_counterfactual_trade40_batch(
            frame, np.asarray([len(days) - 1]), ticker="2330",
            schedule=(period,), exchange_calendar=days,
        )
        self.assertFalse(batch.valid_mask[0]); self.assertTrue(np.isnan(batch.target_raw_r[0]))
        self.assertEqual(batch.reasons[0], "no_future_executable_research_session")

    def test_no_prior_parameter_is_not_backfilled(self):
        days, frame, params, period = fixture()
        formal = (ExecutableMinPeriod(days[50], None, params, "formal_min_roos_seed", "s"),)
        batch = compute_counterfactual_trade40_batch(
            frame, np.asarray([48]), ticker="2330",
            schedule=formal, exchange_calendar=days,
        )
        self.assertFalse(batch.valid_mask[0])
        self.assertEqual(batch.reasons[0], "no_historical_effective_Min_param")


class FormalMinRollingContextCases(unittest.TestCase):
    """Differential witnesses against the real Portfolio loop, without mocks.

    Both paths receive the same hypothetical originating setup. All subsequent
    signals are produced by generate_signals and all fills use canonical core
    execution. Portfolio cash/selector constraints do not bind this one trade.
    """

    @staticmethod
    def _fixture(prices=(105, 106, 100, 102, 103, 104), *, tail_sessions=0):
        days = pd.bdate_range(end="2025-12-30", periods=51).append(
            pd.DatetimeIndex(["2025-12-31", "2026-01-02", "2026-01-05",
                              "2026-01-06", "2026-01-07", "2026-01-08"])
        )
        if tail_sessions:
            days = days.append(pd.bdate_range("2026-01-09", periods=tail_sessions))
        close = np.full(len(days), 100.0)
        close[30:38] = 120.0  # Old long breakout window still contains this high.
        close[51:57] = prices
        close[57:] = prices[-1]
        frame = pd.DataFrame({
            "Open": close.copy(), "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": np.full(len(days), 100_000.0),
        }, index=days)
        frame.attrs["ticker"] = "2330"
        params = V16StrategyParams(
            high_len=30, atr_len=3, atr_buy_tol=0.0,
            atr_times_init=5.0, atr_times_trail=20.0,
            tp_percent=0.0, min_entry_notional=0.0,
            use_breakout_ema_filter=False, use_bb=False, use_kc=False,
            use_vol=False, use_breakout_return_filter=False,
            use_breakout_false_filter=False, use_history_threshold=False,
            use_breakout_quality_filter=False, use_breakout_quality_ranking=False,
        )
        return frame, params

    @staticmethod
    def _schedule(frame, old_params, new_params, kind):
        from core.params_io import params_to_json_dict
        rows = {str(frame.index[0].date()): params_to_json_dict(old_params),
                "2026-01-01": params_to_json_dict(new_params)}
        if kind == "rolling_active_param_ensemble":
            payload = {
                "schema_type": "optimizer_active_param_ensemble", "mode": "rolling",
                "params_ensemble_by_effective_date": {
                    day: [{"member_index": 1, "seed": 1, "params": params}]
                    for day, params in rows.items()
                },
            }
        else:
            payload = {"schema_type": "rolling_oos_param_set",
                       "params_by_effective_date": rows}
        payload["summary"] = {"oos_end_date": str(frame.index[-1].date())}
        records = resolve_formal_min_param_records(kind, payload)
        return build_formal_min_execution_schedule(
            records, seed=1, param_sha256="a" * 64,
        )[0]

    @staticmethod
    def _formal(frame, old_params, new_params, kind, *, cutoff=None):
        from core.portfolio_engine import run_portfolio_timeline
        from core.portfolio_fast_data import (
            pack_static_market_data, pack_optimizer_dynamic_data,
            merge_static_market_with_dynamic, build_normal_setup_index,
            build_trade_stats_index,
        )
        old_cache = tuple(np.copy(value) for value in build_signal_cache(
            frame, old_params, ticker="2330"))
        # The daily Target's only counterfactual is this fresh original intent.
        old_cache[1][50] = True
        old_cache[3][50] = 100.0
        new_cache = build_signal_cache(frame, new_params, ticker="2330")
        static = pack_static_market_data(frame)

        def context(cache):
            fast = {"2330": merge_static_market_with_dynamic(
                static, pack_optimizer_dynamic_data(cache))}
            return {"all_dfs_fast": fast,
                    "all_pit_stats_index": {"2330": build_trade_stats_index([])},
                    "normal_setup_index": build_normal_setup_index(fast)}

        old_context, new_context = context(old_cache), context(new_cache)
        switch = pd.Timestamp("2026-01-01")
        params_for_day = lambda day: new_params if day >= switch else old_params
        context_for_day = lambda day: new_context if day >= switch else old_context
        if kind == "rolling_active_param_ensemble":
            callbacks = {
                "active_param_ensemble_resolver": lambda day: [{
                    "member_index": 1, "member_key": "1", "seed": 1,
                    "params_obj": params_for_day(day),
                    "params_signature": build_portfolio_params_signature(params_for_day(day)),
                }],
                "active_context_ensemble_resolver": lambda day: [context_for_day(day)],
            }
        else:
            callbacks = {"active_params_resolver": params_for_day,
                         "active_context_resolver": context_for_day}
        profile = {"capture_equity_curve": True, "_timing_enabled": False}
        end = frame.index[-1] if cutoff is None else cutoff
        sessions = frame.index[51:][frame.index[51:] <= end].tolist()
        result = run_portfolio_timeline(
            old_context["all_dfs_fast"], {}, sessions, 2025, old_params, 10, False,
            is_training=False, verbose=False, profile_stats=profile,
            pit_stats_index=old_context["all_pit_stats_index"], **callbacks,
        )
        original = [row for row in profile["closed_trade_rows"]
                    if row["signal_date"] == "2025-12-30"]
        return original, result[1]

    @staticmethod
    def _legacy(frame, params):
        return simulate_realized_trade_path_label(
            frame, ticker="2330", signal_pos=50, params=params,
            teacher_effective_date=str(frame.index[0].date()),
            precomputed_signals=build_signal_cache(frame, params, ticker="2330"),
            horizon_end_date=frame.index[-1], counterfactual_entry_limit=100.0,
        )

    def test_pending_rollover_supersedes_original_for_40_60_and_full_cycle(self):
        frame, params = self._fixture(tail_sessions=70)
        frame.loc["2025-12-31", "Open"] = 104.0
        frame.loc["2026-01-02", "Open"] = 105.0
        new_params = replace(params, high_len=3)
        legacy = self._legacy(frame, params)
        self.assertAlmostEqual(legacy.realized_net_r, 0.31658864482054977)
        self.assertEqual(legacy.fill_date, "2026-01-05")
        for kind in ("rolling_oos_param_schedule", "rolling_active_param_ensemble"):
            with self.subTest(kind=kind):
                schedule = self._schedule(frame, params, new_params, kind)
                original, history = self._formal(frame, params, new_params, kind)
                self.assertEqual(original, [])
                # Formal Replay does buy a different origin; its loss is not
                # the counterfactual 12/30 event's NULL outcome.
                fills = history[history["Type"].str.startswith("買進")]
                self.assertEqual(fills.iloc[0]["買訊日"], "2025-12-31")
                self.assertEqual(fills.iloc[0]["Date"], "2026-01-02")
                for horizon in (40, 60, None):
                    actual = compute_counterfactual_trade40_batch(
                        frame, np.asarray([50]), ticker="2330", schedule=schedule,
                        exchange_calendar=frame.index, horizon_sessions=horizon,
                    )
                    self.assertFalse(actual.valid_mask[0])
                    self.assertTrue(np.isnan(actual.target_raw_r[0]))
                    self.assertEqual(actual.reasons, ("no_formal_min_signal_or_continuation",))

    def test_pending_cleanup_consumes_current_indicator_without_replacing_origin_params(self):
        frame, params = self._fixture((105, 108, 104, 105, 100, 104))
        new_params = replace(params, atr_times_trail=0.5)
        self.assertEqual(self._legacy(frame, params).fill_date, "2026-01-07")
        for kind in ("rolling_oos_param_schedule", "rolling_active_param_ensemble"):
            with self.subTest(kind=kind):
                schedule = self._schedule(frame, params, new_params, kind)
                original, _ = self._formal(frame, params, new_params, kind)
                self.assertEqual(original, [])
                actual = compute_counterfactual_trade40_batch(
                    frame, np.asarray([50]), ticker="2330", schedule=schedule,
                    exchange_calendar=frame.index, horizon_sessions=None,
                )
                self.assertTrue(np.isnan(actual.target_raw_r[0]))
                self.assertEqual(actual.reasons, ("no_formal_min_signal_or_continuation",))

    def test_filled_position_uses_current_sell_context_and_origin_numeric_rules(self):
        frame, params = self._fixture((100, 105, 100, 99, 101, 104))
        new_params = replace(params, atr_times_trail=0.5, fixed_risk=0.04)
        self.assertGreater(self._legacy(frame, params).realized_net_r, 0.0)
        for kind in ("rolling_oos_param_schedule", "rolling_active_param_ensemble"):
            with self.subTest(kind=kind):
                schedule = self._schedule(frame, params, new_params, kind)
                original, _ = self._formal(frame, params, new_params, kind)
                self.assertEqual(len(original), 1)
                expected = original[0]
                self.assertEqual(expected["entry_trade_date"], "2025-12-31")
                self.assertEqual(expected["exit_date"], "2026-01-06")
                self.assertEqual(expected["exit_price"], 99.0)
                self.assertEqual(expected["pnl"], -1470.0)
                actual = compute_counterfactual_trade40_batch(
                    frame, np.asarray([50]), ticker="2330", schedule=schedule,
                    exchange_calendar=frame.index, horizon_sessions=None,
                )
                self.assertFalse(actual.valid_mask[0])
                self.assertEqual(actual.reasons, ("no_formal_min_signal_or_continuation",))

    def test_filled_position_trailing_stop_consumes_current_atr(self):
        frame, params = self._fixture((100, 105, 100, 99, 101, 104))
        params = replace(params, atr_len=1, atr_times_trail=2.0)
        new_params = replace(params, atr_len=3)
        for current_params in (params, new_params):
            self.assertFalse(build_signal_cache(frame, current_params, ticker="2330")[2][51:].any())
        self.assertGreater(self._legacy(frame, params).realized_net_r, 0.0)
        for kind in ("rolling_oos_param_schedule", "rolling_active_param_ensemble"):
            with self.subTest(kind=kind):
                schedule = self._schedule(frame, params, new_params, kind)
                original, _ = self._formal(frame, params, new_params, kind)
                self.assertEqual(len(original), 1)
                expected = original[0]
                self.assertEqual(expected["exit_date"], "2026-01-05")
                self.assertEqual(expected["exit_price"], 99.3)
                actual = compute_counterfactual_trade40_batch(
                    frame, np.asarray([50]), ticker="2330", schedule=schedule,
                    exchange_calendar=frame.index, horizon_sessions=40,
                )
                self.assertFalse(actual.valid_mask[0])
                self.assertEqual(actual.reasons, ("no_formal_min_signal_or_continuation",))

    def test_continuation_keeps_origin_entry_geometry_sizing_and_unchanged_path(self):
        frame, params = self._fixture()
        baseline = self._legacy(frame, params)
        for kind in ("rolling_oos_param_schedule", "rolling_active_param_ensemble"):
            for new_params in (params, replace(params, atr_times_init=1.0,
                                               fixed_risk=0.04, atr_buy_tol=10.0)):
                with self.subTest(kind=kind, params_changed=new_params != params):
                    schedule = self._schedule(frame, params, new_params, kind)
                    original, history = self._formal(frame, params, new_params, kind)
                    self.assertEqual(len(original), 1)
                    self.assertEqual(original[0]["entry_trade_date"], "2026-01-05")
                    self.assertEqual(original[0]["entry_price"], 100.0)
                    self.assertEqual(original[0]["pnl"], baseline.realized_net_pnl)
                    self.assertEqual(original[0]["r_mult"], baseline.realized_net_r)
                    fills = history[history["Type"].str.startswith("買進")]
                    self.assertEqual(int(fills.iloc[0]["股數"]), 931)
                    actual = compute_counterfactual_trade40_batch(
                        frame, np.asarray([50]), ticker="2330", schedule=schedule,
                        exchange_calendar=frame.index, horizon_sessions=None,
                    )
                    self.assertFalse(actual.valid_mask[0])
                    self.assertEqual(actual.reasons, ("no_formal_min_signal_or_continuation",))

    def test_no_future_parameter_access_and_midpath_gap_fails_closed(self):
        from filters.breakout_quality import daily_trade40_target as owner
        for prices in ((105, 106, 100, 102, 103, 104), (100, 105, 100, 99, 101, 104)):
            frame, params = self._fixture(prices)
            params = replace(params, high_len=3)
            frame.loc[frame.index[50], "Close"] = 105.0
            frame.loc[frame.index[50], "Open"] = 99.0
            frame.loc[frame.index[50], "High"] = 106.0
            schedule = self._schedule(frame, params, replace(params, high_len=4),
                                      "rolling_oos_param_schedule")
            self.assertTrue(build_signal_cache(frame, params, ticker="2330")[1][50])
            with patch.object(owner, "build_signal_cache", wraps=build_signal_cache) as cache:
                short = compute_counterfactual_trade40_batch(
                    frame, np.asarray([50]), ticker="2330", schedule=schedule,
                    exchange_calendar=frame.index, horizon_sessions=None,
                    research_cutoff="2025-12-31",
                )
                self.assertFalse(short.valid_mask[0])
                self.assertEqual(cache.call_count, 1)
                self.assertIs(cache.call_args.args[1], schedule[0].params)
            # Missing active params are invalid source truth, both while still
            # pending and while holding. The next future period cannot fill it.
            gap_schedule = (schedule[0], replace(schedule[1], start=pd.Timestamp("2026-01-05")))
            with self.assertRaisesRegex(ValueError,
                                        "no_historical_effective_Min_param.*2026-01-02"):
                compute_counterfactual_trade40_batch(
                    frame, np.asarray([50]), ticker="2330", schedule=gap_schedule,
                    exchange_calendar=frame.index, horizon_sessions=None,
                )


class FormalMinSignalStateParityCases(unittest.TestCase):
    """Use genuine price-breakout signals, then compare official single-stock
    Portfolio executions with both raw Target outcomes. No synthetic entry
    limit, portfolio cash hindsight or custom broker formula is introduced.
    """

    @staticmethod
    def _case(*, continuation: bool):
        days, frame, params, period = fixture(count=150)
        if continuation:
            # The original setup at 75 cannot fill at session 76; its pending
            # order fills at 77 using its frozen originating signal geometry.
            frame.loc[days[76], ["Open", "High", "Low", "Close"]] = [151., 153., 147., 150.]
        return days, frame, params, period

    @staticmethod
    def _portfolio(frame, params, dates, start_pos, stop_pos):
        from core.portfolio_engine import run_portfolio_timeline
        from core.portfolio_fast_data import (
            pack_static_market_data, pack_optimizer_dynamic_data,
            merge_static_market_with_dynamic, build_trade_stats_index,
        )
        cache = build_signal_cache(frame, params, ticker="2330")
        static = pack_static_market_data(frame)
        fast = {"2330": merge_static_market_with_dynamic(
            static, pack_optimizer_dynamic_data(cache),
        )}
        captured = {"capture_equity_curve": True, "_timing_enabled": False}
        run_portfolio_timeline(
            fast, {}, dates[start_pos:stop_pos].tolist(), 2023,
            params, 10, False, is_training=False, verbose=False,
            profile_stats=captured,
            pit_stats_index={"2330": build_trade_stats_index([])},
        )
        return list(captured["closed_trade_rows"])

    def test_only_real_min_setup_or_continuation_has_a_trade_target(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, params, period = self._case(continuation=False)
        self.assertTrue(build_signal_cache(frame, params, ticker="2330")[1][75])
        self.assertFalse(build_signal_cache(frame, params, ticker="2330")[1][10])
        with patch.object(owner, "simulate_realized_trade_path_label",
                          wraps=simulate_realized_trade_path_label) as path:
            batch = compute_counterfactual_trade40_batch(
                frame, np.asarray([10, 75, 76, 77]), ticker="2330",
                schedule=(period,), exchange_calendar=days,
                research_cutoff=days[114], horizon_sessions=None,
                include_net_pnl=True,
            )
        self.assertEqual(batch.reasons[0], "no_formal_min_signal_or_continuation")
        self.assertFalse(batch.valid_mask[0]); self.assertTrue(np.isnan(batch.target_raw_r[0]))
        self.assertTrue(np.isnan(batch.target_raw_pnl[0]))
        self.assertTrue(all(batch.valid_mask[1:]))
        self.assertEqual([c.kwargs["signal_pos"] for c in path.call_args_list], [75, 75, 75])
        self.assertEqual([c.kwargs["decision_pos"] for c in path.call_args_list], [75, 76, 77])
        limits = [c.kwargs["starting_signal_state"]["orig_limit"] for c in path.call_args_list]
        self.assertEqual(limits, [limits[0]] * 3)
        self.assertIsNone(path.call_args_list[0].kwargs["starting_signal_state"]["shadow_position"])
        self.assertIsNotNone(path.call_args_list[1].kwargs["starting_signal_state"]["shadow_position"])

    def test_normal_and_continuation_portfolio_ledger_match_target_pnl_and_r(self):
        for continued in (False, True):
            with self.subTest(continuation=continued):
                days, frame, params, period = self._case(continuation=continued)
                decision = 76 if continued else 75
                batch = compute_counterfactual_trade40_batch(
                    frame, np.asarray([decision]), ticker="2330",
                    schedule=(period,), exchange_calendar=days,
                    research_cutoff=days[114], horizon_sessions=None,
                    include_net_pnl=True,
                )
                self.assertTrue(batch.valid_mask[0])
                trades = self._portfolio(frame, params, days, 76, 115)
                self.assertTrue(trades)
                actual = trades[0]
                self.assertEqual(actual["signal_date"], str(days[75].date()))
                self.assertEqual(actual["entry_type"], "extended" if continued else "normal")
                self.assertAlmostEqual(float(batch.target_raw_pnl[0]), actual["pnl"], places=7)
                self.assertAlmostEqual(float(batch.target_raw_r[0]), actual["r_mult"], places=6)
                self.assertEqual(str(batch.maturity_dates[0]), actual["exit_date"])

    def test_continuation_keeps_origin_p16_even_after_effective_date_switch(self):
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, params, _period = self._case(continuation=True)
        changed = replace(params, fixed_risk=0.07, atr_buy_tol=8.0,
                          atr_times_init=1.0, high_len=11)
        old = ExecutableMinPeriod(days[0], days[76], params, "formal_min_roos_seed",
                                  build_portfolio_params_signature(params))
        new = ExecutableMinPeriod(days[77], None, changed, "formal_min_roos_seed",
                                  build_portfolio_params_signature(changed))
        with patch.object(owner, "simulate_realized_trade_path_label",
                          wraps=simulate_realized_trade_path_label) as path:
            batch = compute_counterfactual_trade40_batch(
                frame, np.asarray([76]), ticker="2330",
                schedule=(old, new), exchange_calendar=days,
                research_cutoff=days[114], horizon_sessions=None,
                include_net_pnl=True,
            )
        self.assertEqual(len(path.call_args_list), 1)
        self.assertIs(path.call_args.kwargs["params"], params)
        self.assertIs(path.call_args.kwargs["starting_signal_state"]["_params_obj"], params)
        self.assertEqual(path.call_args.kwargs["signal_pos"], 75)
        self.assertTrue(batch.valid_mask[0])

    def test_resume_rejects_invented_order_and_wrong_source_identity(self):
        from core.extended_signals import create_signal_tracking_state
        days, frame, params, _period = self._case(continuation=True)
        cache = build_signal_cache(frame, params, ticker="2330")
        state = create_signal_tracking_state(float(cache[3][75]), float(cache[0][75]),
                                             params, ticker="2330", signal_date=days[75])
        kw = dict(stock_df=frame, ticker="2330", signal_pos=75,
                  params=params, teacher_effective_date=str(days[0].date()),
                  precomputed_signals=cache, horizon_end_date=days[114],
                  decision_pos=76, starting_signal_state=state)
        other = dict(state, ticker="9999")
        with self.assertRaisesRegex(ValueError, "canonical signal"):
            simulate_realized_trade_path_label(**{**kw, "starting_signal_state": other})
        with self.assertRaisesRegex(ValueError, "hypothetical fresh limit"):
            simulate_realized_trade_path_label(**{**kw, "counterfactual_entry_limit": 106.})
        with self.assertRaisesRegex(ValueError, "requires the canonical"):
            simulate_realized_trade_path_label(**{**kw, "starting_signal_state": None})




class FormalMinTrade40TrainingSourcePreflightCases(unittest.TestCase):
    """The new supervised target must fail closed before selecting old teachers."""

    @staticmethod
    def _source(*, start="2018-01-02", publish=None, seed=693545351):
        from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITSource
        days, frame, params, _period = fixture(count=140, start=start)
        period = ExecutableMinPeriod(
            pd.Timestamp(days[0]), None, params, "formal_min_roos_seed",
            build_portfolio_params_signature(params),
        )
        return days, FormalMinTrade40PITSource(
            seed=seed, param_sha256="a" * 64, schedule=(period,),
            publication_dates=(str(publish or days[0].date()),),
        )

    def test_certified_historical_seed_schedule_can_pass_preflight(self):
        from services.breakout_quality.formal_mintrade40_pit_source import verify_formal_mintrade40_pit_source
        days, source = self._source()
        evidence = verify_formal_mintrade40_pit_source(
            source, expected_seed=source.seed, information_dates=[days[70]],
            executable_dates=[days[71]], fit_cutoff=days[110],
        )
        self.assertEqual(evidence["sample_count"], 1)
        self.assertEqual(evidence["validation"], "PIT_SCHEDULE_PREFLIGHT_ONLY_NOT_FIT_AUTHORIZATION")

    def test_current_2021_p16_cannot_label_historical_training_rows(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITBlocked, verify_formal_mintrade40_pit_source,
        )
        days, source = self._source(start="2021-01-04")
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "Future-dated Min schedule"):
            verify_formal_mintrade40_pit_source(
                source, expected_seed=source.seed,
                information_dates=[pd.Timestamp("2019-01-03")],
                executable_dates=[pd.Timestamp("2019-01-04")],
                fit_cutoff=pd.Timestamp("2020-12-31"),
            )

    def test_teacher_seed_mismatch_rejected(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITBlocked, verify_formal_mintrade40_pit_source,
        )
        days, source = self._source()
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "Seed-specific"):
            verify_formal_mintrade40_pit_source(
                source, expected_seed=2014432738,
                information_dates=[days[50]], executable_dates=[days[51]],
                fit_cutoff=days[110],
            )

    def test_future_published_teacher_rejected(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITBlocked, verify_formal_mintrade40_pit_source,
        )
        days, source = self._source(publish="2020-01-01")
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "backfilled"):
            verify_formal_mintrade40_pit_source(
                source, expected_seed=source.seed,
                information_dates=[days[60]], executable_dates=[days[61]],
                fit_cutoff=days[110],
            )

    def test_post_cutoff_rows_and_retired_teacher_rejected(self):
        from dataclasses import replace as dataclass_replace
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITBlocked, verify_formal_mintrade40_pit_source,
        )
        days, source = self._source()
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "Post-FIT-cutoff"):
            verify_formal_mintrade40_pit_source(
                source, expected_seed=source.seed,
                information_dates=[days[60]], executable_dates=[days[61]],
                fit_cutoff=days[60],
            )
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "Retired or nonformal"):
            verify_formal_mintrade40_pit_source(
                dataclass_replace(source, source_id="retired_p2_teacher"),
                expected_seed=source.seed, information_dates=[days[60]],
                executable_dates=[days[61]], fit_cutoff=days[110],
            )


    def test_mature_labels_match_canonical_and_immature_cannot_force_closeout(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            materialize_formal_mintrade40_pit_labels,
        )
        days, source = self._source()
        _, frame, _, _ = fixture(count=140, start="2018-01-02")
        fit = days[110]
        labels = materialize_formal_mintrade40_pit_labels(
            source, expected_seed=source.seed, frame=frame, positions=np.array([70, 90]),
            ticker="2330", exchange_calendar=days, fit_cutoff=fit,
        )
        canonical = compute_counterfactual_trade40_batch(
            frame.loc[:fit], np.array([70]), ticker="2330", schedule=source.schedule,
            exchange_calendar=days[days <= fit], research_cutoff=fit,
        )
        self.assertEqual(labels.matured.tolist(), [True, False])
        self.assertEqual(labels.completed.tolist(), [True, False])
        self.assertAlmostEqual(float(labels.raw_net_r[0]), float(canonical.target_raw_r[0]))
        self.assertTrue(np.isnan(labels.raw_net_r[1]))
        self.assertTrue(np.isnat(labels.maturity_dates[1]))

    def test_future_prices_do_not_change_fit_cutoff_labels(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            materialize_formal_mintrade40_pit_labels,
        )
        days, source = self._source()
        _, frame, _, _ = fixture(count=140, start="2018-01-02")
        kwargs = dict(source=source, expected_seed=source.seed, positions=np.array([70, 90]),
                      ticker="2330", exchange_calendar=days, fit_cutoff=days[110])
        before = materialize_formal_mintrade40_pit_labels(frame=frame, **kwargs)
        tampered = frame.copy()
        tampered.loc[tampered.index > days[110], ["Open", "High", "Low", "Close"]] *= 15
        after = materialize_formal_mintrade40_pit_labels(frame=tampered, **kwargs)
        np.testing.assert_equal(before.raw_net_r, after.raw_net_r)
        np.testing.assert_array_equal(before.matured, after.matured)

    def test_natural_early_maturity_is_accepted_without_full_horizon(self):
        from services.breakout_quality import formal_mintrade40_pit_source as owner
        days, source = self._source()
        _, frame, _, _ = fixture(count=140, start="2018-01-02")
        early = SimpleNamespace(
            target_raw_r=np.array([1.25], dtype=np.float32),
            valid_mask=np.array([True]),
            maturity_dates=np.array([days[88].to_datetime64().astype("datetime64[D]")]),
        )
        with patch.object(owner, "compute_counterfactual_trade40_batch", return_value=early):
            labels = owner.materialize_formal_mintrade40_pit_labels(
                source, expected_seed=source.seed, frame=frame,
                positions=np.array([75]), ticker="2330", exchange_calendar=days,
                fit_cutoff=days[95],
            )
        self.assertTrue(labels.completed[0])
        self.assertEqual(float(labels.raw_net_r[0]), 1.25)



class FormalMinTrade40OptimizerAttestationCases(unittest.TestCase):
    """Source adapter must accept only a genuine-shape, as-of Min Optimizer lineage."""

    @staticmethod
    def _artifacts(directory, *, seed=693545351):
        import json
        from core.params_io import params_to_json_dict
        from core.file_integrity import canonical_json_sha256
        from services.optimizer.formal_mintrade40_teacher_source import (
            HISTORICAL_MINTRADE40_TRAINING_SET,
        )
        from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
        from services.optimizer.strategy_param_training import (
            MIN_ROOS_SEARCH_FIELDS, _build_min_roos_schedule_contract,
        )
        root = (Path(directory) / "outputs" / "optimizer" / "mintrade40_label_history"
                / f"seed_{seed}" / "p2_dl_off_trained" / "active_params")
        root.mkdir(parents=True)
        policy = _canonical_schedule_build_contract(family="min")
        params = params_to_json_dict(V16StrategyParams(
            high_len=10, atr_len=5, use_breakout_quality_filter=False,
            use_breakout_quality_ranking=False,
        ))
        effective_dates = pd.date_range("2014-01-01", periods=7, freq="YS")
        folds = []
        members = {}
        for effective in effective_dates:
            oos_end = effective + pd.DateOffset(years=1) - pd.Timedelta(days=1)
            folds.append({
                "oos_start_date": str(effective.date()),
                "oos_end_date": str(oos_end.date()),
                "selection_start_date": str((effective - pd.DateOffset(months=policy["train_window_months"])).date()),
                "selection_end_date": str((effective - pd.Timedelta(days=1)).date()),
            })
            members[str(effective.date())] = [{"seed": seed, "params": params}]
        payload = {
            "schema_type": "optimizer_active_param_ensemble", "mode": "rolling",
            "breakout_quality_param_adaptation": {
                "mode": "min_roos_training",
                "parameter_set": HISTORICAL_MINTRADE40_TRAINING_SET,
                "training_dl_enabled": False,
                "fixed_rule_contract": "all_rule_filters_off",
                "search_fields": list(MIN_ROOS_SEARCH_FIELDS),
            },
            "meta": {
                "first_oos_date": "2014-01-01", "last_oos_date": "2020-12-31",
                "train_window_months": policy["train_window_months"],
                "oos_horizon_months": policy["oos_horizon_months"],
                "trials_per_fold": policy["trials_per_fold"],
            },
            "params_ensemble_by_effective_date": members,
            "folds": folds,
            "summary": {"folds": len(folds)},
        }
        reference = _build_min_roos_schedule_contract(
            first_oos_date="2014-01-01", last_oos_date="2020-12-31",
            train_window_months=policy["train_window_months"],
            oos_months=policy["oos_horizon_months"],
        )
        contract = {
            "mode": "binary_dl_filter_min_roos_param_adaptation",
            "arm_id": HISTORICAL_MINTRADE40_TRAINING_SET,
            "training_dl_enabled": False, "dataset": "full",
            "dataset_identity": {"csv_inventory_sha256": "b" * 64},
            "parameter_reference": {
                "type": reference["reference_type"], "path": None,
                "sha256": reference["sha256"],
            },
            "rolling_policy": dict(reference["meta"]),
            "search_fields": list(MIN_ROOS_SEARCH_FIELDS),
            "trials_per_fold": policy["trials_per_fold"],
            "binary_pit": None, "binary_runtime": None,
        }
        identity = canonical_json_sha256(contract)
        preflight = {**contract, "runtime_identity_sha256": identity,
                     "status": "PREFLIGHT_PASS"}
        summary = {"status": "COMPLETED",
                   "parameter_set": HISTORICAL_MINTRADE40_TRAINING_SET,
                   "runtime_identity_sha256": identity}
        source = root / "roos_base_best.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        (root.parent / "rolling_preflight.json").write_text(json.dumps(preflight), encoding="utf-8")
        (root.parent / "rolling_optimizer_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return source, payload, preflight, summary

    @staticmethod
    def _rewrite(path, payload):
        import json
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    def test_optimizer_adapter_preserves_seed_and_immutable_source_sha(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        from core.strategy_param_artifacts import compute_strategy_param_file_sha256
        with TemporaryDirectory() as temp:
            source, *_ = self._artifacts(temp)
            identity = load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)
            self.assertEqual(identity.seed, 693545351)
            self.assertEqual(identity.param_sha256, compute_strategy_param_file_sha256(source))
            self.assertEqual(len(identity.schedule), 7)
            self.assertEqual(identity.publication_dates[0], "2013-12-31")
            self.assertEqual(str(identity.schedule[-1].end.date()), "2020-12-31")

    def test_optimizer_adapter_rejects_old_p2_teacher(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as temp:
            source, payload, *_ = self._artifacts(temp)
            payload["breakout_quality_param_adaptation"]["parameter_set"] = "P2_HISTORY"
            self._rewrite(source, payload)
            with self.assertRaisesRegex(ValueError, "policy/provenance"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

    def test_optimizer_adapter_rejects_mixed_seed(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as temp:
            source, payload, *_ = self._artifacts(temp)
            payload["params_ensemble_by_effective_date"]["2015-01-01"][0]["seed"] = 2014432738
            self._rewrite(source, payload)
            with self.assertRaisesRegex(ValueError, "Seed-specific"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

    def test_optimizer_adapter_rejects_as_of_or_history_shortening(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as temp:
            source, payload, *_ = self._artifacts(temp)
            payload["folds"][0]["selection_start_date"] = "2011-01-01"
            self._rewrite(source, payload)
            with self.assertRaisesRegex(ValueError, "not PIT-aligned"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

    def test_optimizer_adapter_rejects_missing_fold_or_fake_preflight_identity(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as temp:
            source, payload, preflight, _ = self._artifacts(temp)
            payload["folds"].pop()
            self._rewrite(source, payload)
            with self.assertRaisesRegex(ValueError, "provenance incomplete"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)
            payload["folds"].append({
                "oos_start_date": "2020-01-01", "oos_end_date": "2020-12-31",
                "selection_start_date": "2010-01-01", "selection_end_date": "2019-12-31",
            })
            self._rewrite(source, payload)
            preflight["dataset_identity"]["csv_inventory_sha256"] = "c" * 64
            self._rewrite(source.parent.parent / "rolling_preflight.json", preflight)
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)


class FormalMinTrade40OptimizerBuildGuardCases(unittest.TestCase):
    """Historical P16 is built ONLY via canonical optimizer, never a P2 recovery."""

    def test_historical_build_requires_explicit_authorization(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            prepare_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as root:
            with patch(
                "services.optimizer.strategy_param_training.prepare_selection_historical_p2_params",
                side_effect=AssertionError("Optimizer must not execute"),
            ) as producer:
                with self.assertRaisesRegex(ValueError, "requires model-work authorization"):
                    prepare_formal_mintrade40_optimizer_teacher(
                        project_root=root, seed=693545351,
                    )
                producer.assert_not_called()

    def test_producer_gets_formal_120_month_window_and_isolated_physical_path(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            HISTORICAL_MINTRADE40_TRAINING_SET,
            prepare_formal_mintrade40_optimizer_teacher,
        )
        from filters.breakout_quality.strategy_compare_sources import (
            PARAM_POLICY_SPECS, PARAM_POLICY_BASE_FINALIST_BEST,
        )
        from services.optimizer.strategy_param_service import _canonical_schedule_build_contract
        with TemporaryDirectory() as root:
            seed = 693545351
            expected = (Path(root) / "outputs" / "optimizer"
                        / "mintrade40_label_history" / f"seed_{seed}"
                        / "p2_dl_off_trained" / "active_params"
                        / PARAM_POLICY_SPECS[PARAM_POLICY_BASE_FINALIST_BEST]["filename"])
            sentinel = object()
            with (patch(
                    "services.optimizer.strategy_param_training.prepare_selection_historical_p2_params",
                    return_value={"params_path": expected},
                  ) as producer,
                  patch(
                    "services.optimizer.formal_mintrade40_teacher_source.load_formal_mintrade40_optimizer_teacher",
                    return_value=sentinel,
                  ) as loader):
                actual = prepare_formal_mintrade40_optimizer_teacher(
                    project_root=root, seed=seed, authorized=True,
                )
            self.assertIs(actual, sentinel)
            loader.assert_called_once_with(expected.resolve(), expected_seed=seed)
            kwargs = producer.call_args.kwargs
            policy = _canonical_schedule_build_contract(family="min")
            self.assertEqual(kwargs["parameter_set_identity"], HISTORICAL_MINTRADE40_TRAINING_SET)
            self.assertEqual(kwargs["first_oos_date"], "2014-01-01")
            self.assertEqual(kwargs["last_oos_date"], "2020-12-31")
            self.assertEqual(kwargs["train_window_months"], int(policy["train_window_months"]))
            self.assertEqual(kwargs["oos_months"], int(policy["oos_horizon_months"]))
            self.assertEqual(kwargs["trials_per_fold"], int(policy["trials_per_fold"]))
            self.assertEqual(kwargs["optimizer_seed"], seed)
            self.assertEqual(kwargs["dataset"], "full")
            self.assertEqual(kwargs["output_relative_dir"], expected.parent.parent.parent)
            self.assertFalse(kwargs["recover_completed_strategy_compare"])
            self.assertIsNone(kwargs.get("seed_ensemble_policy_override"))

    def test_stale_existing_source_blocks_and_does_not_run_optimizer(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            prepare_formal_mintrade40_optimizer_teacher,
        )
        from filters.breakout_quality.strategy_compare_sources import (
            PARAM_POLICY_SPECS, PARAM_POLICY_BASE_FINALIST_BEST,
        )
        with TemporaryDirectory() as root:
            seed = 693545351
            path = (Path(root) / "outputs" / "optimizer" / "mintrade40_label_history"
                    / f"seed_{seed}" / "p2_dl_off_trained" / "active_params"
                    / PARAM_POLICY_SPECS[PARAM_POLICY_BASE_FINALIST_BEST]["filename"])
            path.parent.mkdir(parents=True)
            path.write_text("{}", encoding="utf-8")
            with patch(
                "services.optimizer.strategy_param_training.prepare_selection_historical_p2_params",
                side_effect=AssertionError("Must preserve existing corrupt source"),
            ) as producer:
                with self.assertRaises(ValueError):
                    prepare_formal_mintrade40_optimizer_teacher(
                        project_root=root, seed=seed, authorized=True,
                    )
                producer.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "{}")

    def test_optimizer_adapter_rejects_corrupt_raw_inventory_sha(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        from core.file_integrity import canonical_json_sha256
        with TemporaryDirectory() as root:
            source, _, preflight, summary = FormalMinTrade40OptimizerAttestationCases._artifacts(root)
            preflight["dataset_identity"]["csv_inventory_sha256"] = "non-sha256"
            contract = {key: value for key, value in preflight.items()
                        if key not in {"status", "runtime_identity_sha256"}}
            fingerprint = canonical_json_sha256(contract)
            preflight["runtime_identity_sha256"] = fingerprint
            summary["runtime_identity_sha256"] = fingerprint
            FormalMinTrade40OptimizerAttestationCases._rewrite(
                source.parent.parent / "rolling_preflight.json", preflight)
            FormalMinTrade40OptimizerAttestationCases._rewrite(
                source.parent.parent / "rolling_optimizer_summary.json", summary)
            with self.assertRaisesRegex(ValueError, "inventory SHA256 invalid"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

    def test_optimizer_adapter_rejects_wrong_historical_reference_even_when_rehashed(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        from core.file_integrity import canonical_json_sha256
        with TemporaryDirectory() as root:
            source, _, preflight, summary = FormalMinTrade40OptimizerAttestationCases._artifacts(root)
            preflight["parameter_reference"]["sha256"] = "c" * 64
            contract = {key: value for key, value in preflight.items()
                        if key not in {"status", "runtime_identity_sha256"}}
            fingerprint = canonical_json_sha256(contract)
            preflight["runtime_identity_sha256"] = fingerprint
            summary["runtime_identity_sha256"] = fingerprint
            FormalMinTrade40OptimizerAttestationCases._rewrite(
                source.parent.parent / "rolling_preflight.json", preflight)
            FormalMinTrade40OptimizerAttestationCases._rewrite(
                source.parent.parent / "rolling_optimizer_summary.json", summary)
            with self.assertRaisesRegex(ValueError, "reference identity invalid"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

    def test_optimizer_adapter_rejects_mixed_fold_optimizer_seed(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_teacher,
        )
        with TemporaryDirectory() as root:
            source, payload, *_ = FormalMinTrade40OptimizerAttestationCases._artifacts(root)
            payload["folds"][0]["optimizer_seed"] = 2014432738
            FormalMinTrade40OptimizerAttestationCases._rewrite(source, payload)
            with self.assertRaisesRegex(ValueError, "Seed-specific"):
                load_formal_mintrade40_optimizer_teacher(source, expected_seed=693545351)

if __name__ == "__main__":
    unittest.main()
