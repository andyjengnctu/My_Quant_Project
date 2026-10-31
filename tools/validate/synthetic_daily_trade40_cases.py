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
    def test_replayed_trade_dates_reuse_verified_parameter_period(self):
        # The same decision/executable day can be revisited by many overlapping
        # counterfactual trades. Parameter lookup is execution-only; neither
        # active schedule nor replayed net R/NULL may change.
        from filters.breakout_quality import daily_trade40_target as owner
        days, frame, _params, period = fixture(count=150)
        positions = np.arange(45, 118, dtype=np.int64)
        with patch.object(owner, "resolve_historical_min_period",
                          wraps=resolve_historical_min_period) as lookup:
            cached = compute_counterfactual_trade40_batch(
                frame, positions, ticker="2330", schedule=(period,),
                exchange_calendar=days, research_cutoff=days[130],
            )
        baseline = compute_counterfactual_trade40_batch(
            frame, positions, ticker="2330", schedule=(period,),
            exchange_calendar=days, research_cutoff=days[130],
        )
        self.assertLessEqual(lookup.call_count, len(days))
        self.assertGreater(int(cached.valid_mask.sum()), 0)
        np.testing.assert_array_equal(cached.target_raw_r, baseline.target_raw_r)
        np.testing.assert_array_equal(cached.maturity_dates, baseline.maturity_dates)
        self.assertEqual(cached.reasons, baseline.reasons)

    def test_common_price_truth_independent_of_null_target_subset(self):
        from filters.breakout_quality.daily_ranker_data import compute_daily_opportunity_target_batch
        from filters.breakout_quality.continuous_target import (
            DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID, StrategyAlignedContinuousTargetSpec,
        )
        from filters.breakout_quality.contract import DEFAULT_LABEL_POLICY
        _days, frame, _params, _period = fixture(count=150)
        spec = StrategyAlignedContinuousTargetSpec.from_label_policy(
            replace(DEFAULT_LABEL_POLICY, label_horizon_bars=40),
        )
        positions = np.arange(48, 106, dtype=np.int64)
        is_null = positions % 3 == 0
        all_prices = compute_daily_opportunity_target_batch(
            frame, positions, spec=spec, target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        )
        null_prices = compute_daily_opportunity_target_batch(
            frame, positions[is_null], spec=spec,
            target_id=DAILY_FULL_HORIZON_PURE_MFE_TARGET_ID,
        )
        for column in (
            "valid_mask", "favorable_return", "adverse_return_to_peak",
            "opportunity_bar", "first_risk_breach_bar", "minimum_low_return",
        ):
            np.testing.assert_array_equal(
                np.asarray(getattr(all_prices, column))[is_null],
                np.asarray(getattr(null_prices, column)),
            )

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


    def test_asof_teacher_slicing_excludes_later_fold_and_caps_current_period(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            slice_formal_mintrade40_pit_source_for_fold,
        )
        from dataclasses import replace as dataclass_replace
        days, source = self._source()
        _, _, params, _ = fixture(count=140, start="2018-01-02")
        old = dataclass_replace(source.schedule[0], end=pd.Timestamp("2018-12-31"))
        future = ExecutableMinPeriod(
            pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31"),
            dataclass_replace(params, atr_buy_tol=0.4),
            "formal_min_roos_seed",
            build_portfolio_params_signature(dataclass_replace(params, atr_buy_tol=0.4)),
        )
        full = dataclass_replace(source, schedule=(old, future),
                                 publication_dates=("2018-01-02", "2018-12-31"))
        snapshot, digest = slice_formal_mintrade40_pit_source_for_fold(
            full, fit_cutoff=days[110],
        )
        self.assertEqual(len(snapshot.schedule), 1)
        self.assertEqual(snapshot.param_sha256, full.param_sha256)
        self.assertEqual(snapshot.schedule[0].end, days[110])
        self.assertEqual(snapshot.publication_dates, ("2018-01-02",))
        self.assertEqual(len(digest), 64)
        changed = dataclass_replace(future,
            params=dataclass_replace(future.params, atr_buy_tol=2.2),
            params_signature=build_portfolio_params_signature(
                dataclass_replace(future.params, atr_buy_tol=2.2)))
        revised, revised_digest = slice_formal_mintrade40_pit_source_for_fold(
            dataclass_replace(full, param_sha256="b" * 64, schedule=(old, changed)),
            fit_cutoff=days[110],
        )
        self.assertEqual(digest, revised_digest)
        self.assertEqual(snapshot.schedule, revised.schedule)
        later, later_digest = slice_formal_mintrade40_pit_source_for_fold(
            full, fit_cutoff=pd.Timestamp("2019-06-30"),
        )
        self.assertEqual(len(later.schedule), 2)
        self.assertNotEqual(digest, later_digest)
        self.assertEqual(later.schedule[-1].end, pd.Timestamp("2019-06-30"))

    def test_asof_rejects_missing_history_future_publication_and_bad_identity(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITBlocked, slice_formal_mintrade40_pit_source_for_fold,
        )
        from dataclasses import replace as dataclass_replace
        days, source = self._source()
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "No historical"):
            slice_formal_mintrade40_pit_source_for_fold(
                source, fit_cutoff=pd.Timestamp("2017-12-29"),
            )
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "publication"):
            slice_formal_mintrade40_pit_source_for_fold(
                dataclass_replace(source, publication_dates=("2019-01-01",)),
                fit_cutoff=days[110],
            )
        with self.assertRaisesRegex(FormalMinTrade40PITBlocked, "source identity"):
            slice_formal_mintrade40_pit_source_for_fold(
                dataclass_replace(source, source_id="retired_p2_teacher"),
                fit_cutoff=days[110],
            )

    def test_label_materializer_reuses_only_fold_asof_teacher(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            materialize_formal_mintrade40_pit_labels,
        )
        from dataclasses import replace as dataclass_replace
        days, source = self._source()
        _, frame, params, _ = fixture(count=140, start="2018-01-02")
        cutoff = days[110]
        later = ExecutableMinPeriod(
            pd.Timestamp("2019-01-01"), None,
            dataclass_replace(params, atr_buy_tol=0.7), "formal_min_roos_seed",
            build_portfolio_params_signature(dataclass_replace(params, atr_buy_tol=0.7)),
        )
        earlier = dataclass_replace(source.schedule[0], end=pd.Timestamp("2018-12-31"))
        full = dataclass_replace(source, schedule=(earlier, later),
                                 publication_dates=("2018-01-02", "2018-12-31"))
        args = dict(expected_seed=source.seed, frame=frame,
                    positions=np.array([70, 90]), ticker="2330",
                    exchange_calendar=days, fit_cutoff=cutoff)
        from_full = materialize_formal_mintrade40_pit_labels(full, **args)
        from_prefix = materialize_formal_mintrade40_pit_labels(source, **args)
        np.testing.assert_equal(from_full.raw_net_r, from_prefix.raw_net_r)
        np.testing.assert_array_equal(from_full.matured, from_prefix.matured)
        self.assertEqual(from_full.preflight["fold_teacher_as_of_sha256"],
                         from_prefix.preflight["fold_teacher_as_of_sha256"])
        self.assertFalse(from_full.matured[1])

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
            research_cutoff_forced_mask=np.array([False]),
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

    def test_optimizer_fold_loader_attests_full_source_then_slices_asof(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            load_formal_mintrade40_optimizer_fold_teacher,
        )
        with TemporaryDirectory() as temp:
            path, *_ = self._artifacts(temp)
            snapshot, digest = load_formal_mintrade40_optimizer_fold_teacher(
                path, expected_seed=693545351, fit_cutoff="2016-06-30",
            )
            self.assertEqual(len(snapshot.schedule), 3)
            self.assertEqual(str(snapshot.schedule[-1].end.date()), "2016-06-30")
            self.assertEqual(snapshot.publication_dates,
                             ("2013-12-31", "2014-12-31", "2015-12-31"))
            self.assertEqual(len(digest), 64)
            with self.assertRaisesRegex(ValueError, "No historical"):
                load_formal_mintrade40_optimizer_fold_teacher(
                    path, expected_seed=693545351, fit_cutoff="2013-06-30",
                )
            with self.assertRaisesRegex(ValueError, "Seed-specific|path/seed"):
                load_formal_mintrade40_optimizer_fold_teacher(
                    path, expected_seed=2014432738, fit_cutoff="2016-06-30",
                )

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

class FormalMinTrade40IntegratedResearchCases(unittest.TestCase):
    """New model/strategy capability, with zero actual FIT or market Replay."""

    def test_current_model_exact_and_direct_are_one_formal_pit_source(self):
        from core.breakout_quality_policy import get_breakout_quality_model_test_settings
        from core.strategy_compare_policy import get_strategy_comparison_settings
        from core.breakout_quality_runtime import DAILY_FORMAL_MINTRADE40_NETR_TARGET_ID
        from services.research.target_dl_overview import collect_target_dl_overview
        profile = 'daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        self.assertIn(('MR-13CM', profile), get_breakout_quality_model_test_settings().model_profiles)
        self.assertEqual(get_continuous_ranker_research_spec(profile).model_research_id, 'MR-13CM')
        self.assertEqual(get_continuous_ranker_research_spec(profile).selection_pit_authorized, True)
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        recipe = get_continuous_ranker_execution_recipe(profile)
        self.assertEqual(recipe.continuous_target_id, DAILY_FORMAL_MINTRADE40_NETR_TARGET_ID)
        self.assertEqual(recipe.target_policy.contract_kind, 'formal_seed_mintrade40_netr')
        for mode in ('broker_cash_dl_conversion_oos', 'broker_cash_dl_conversion_rolling'):
            cfg = get_strategy_comparison_settings(mode)
            a, b = cfg.arms['C119'], cfg.arms['C120']
            self.assertEqual(a.dl_id, b.dl_id)
            self.assertEqual(a.dl_id, 'CONT13CM_ROLL')
            self.assertEqual(a.param_source, b.param_source)
            self.assertEqual(a.param_policy, b.param_policy)
            self.assertEqual(a.rule_policy, b.rule_policy)
            self.assertNotEqual(a.dl_runtime_mode, b.dl_runtime_mode)
        rows = collect_target_dl_overview(Path(__file__).resolve().parents[2])['rows']
        assigned = {row['target']: row for row in rows if row['model']=='MR-13CM'}
        self.assertEqual(assigned['DailyUniversal_MinTrade40_NetR']['strategy'],'C119')
        self.assertEqual(assigned['DailyUniversal_MinTrade40_NetR_Direct']['strategy'],'C120')
        self.assertEqual(len(assigned),2)

    def test_missing_original_p16_is_delegated_to_canonical_optimizer_first(self):
        from services.breakout_quality import formal_mintrade40_training_source as owner
        from services.breakout_quality.formal_mintrade40_pit_source import FormalMinTrade40PITBlocked
        with TemporaryDirectory() as root, patch.object(
            owner, 'prepare_formal_mintrade40_optimizer_teacher'
        ) as historical, patch.object(
            owner, 'ensure_fixed_multiseed_strategy_parameter_artifact',
            side_effect=RuntimeError('canonical current P16 producer invoked'),
        ) as current:
            with self.assertRaisesRegex(RuntimeError, 'canonical current P16 producer invoked'):
                owner.load_formal_mintrade40_training_source(
                    project_root=root, seed=42, allow_optimizer_build=True,
                )
            current.assert_called_once()
            historical.assert_not_called()
            current.reset_mock()
            with self.assertRaises(FormalMinTrade40PITBlocked):
                owner.load_formal_mintrade40_training_source(
                    project_root=root, seed=42, allow_optimizer_build=False,
                )
            current.assert_not_called()
            historical.assert_not_called()

    def test_original_optimizer_fold_cutoff_and_seed_are_mandatory(self):
        from services.breakout_quality.formal_mintrade40_training_source import (
            load_formal_mintrade40_training_source,
        )
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITSource, FormalMinTrade40PITBlocked,
        )
        days, _frame, _params, period = fixture()
        historical = replace(period, start=pd.Timestamp('2014-01-01'), end=pd.Timestamp('2020-12-31'))
        current = replace(period, start=pd.Timestamp('2021-01-01'), end=pd.Timestamp('2021-12-31'))
        prior_source = FormalMinTrade40PITSource(
            seed=42, param_sha256='a'*64, schedule=(historical,),
            publication_dates=('2013-12-31',),
        )
        fold = {
            'oos_start_date': '2021-01-01',
            'selection_start_date': '2011-01-01',
            'selection_end_date': '2020-12-31',
            'optimizer_seed': 42,
        }
        from contextlib import ExitStack
        f='services.breakout_quality.formal_mintrade40_training_source.'
        with TemporaryDirectory() as root, ExitStack() as stack:
            stack.enter_context(patch(f+'_current_p16_seed_artifact',return_value=(Path(root)/'p16.json','2021-01-01','2026-03-02','base-finalist-best')))
            stack.enter_context(patch(f+'load_formal_mintrade40_optimizer_teacher',return_value=prior_source))
            payload={'kind':'rolling_active_param_ensemble','payload':{'folds':[fold]}}
            stack.enter_context(patch(f+'load_strategy_param_evaluation_view',return_value=(payload,'b'*64)))
            stack.enter_context(patch(f+'resolve_formal_min_param_records',return_value=({'members':({'seed':42},)},)))
            stack.enter_context(patch(f+'build_formal_min_execution_schedule',return_value=((current,),{})))
            # _validate_requested_param_policy is imported within the owner.
            stack.enter_context(patch('filters.breakout_quality.strategy_compare_sources._validate_requested_param_policy'))
            verified=load_formal_mintrade40_training_source(project_root=root,seed=42)
            self.assertEqual(verified.seed,42)
            self.assertEqual(verified.publication_dates[-1],'2020-12-31')
            fold['optimizer_seed']=2014432738
            with self.assertRaisesRegex(FormalMinTrade40PITBlocked,'Seed'):
                load_formal_mintrade40_training_source(project_root=root,seed=42)
            fold['optimizer_seed']=42
            fold['selection_end_date']='2021-01-10'
            with self.assertRaisesRegex(FormalMinTrade40PITBlocked,'PIT aligned'):
                load_formal_mintrade40_training_source(project_root=root,seed=42)



class FormalMinTrade40EndToEndDefinitionCases(unittest.TestCase):
    """Six real boundary/lineage regression cases; no Oracle caches or FIT run."""

    def test_oos_signal_boundary_is_identical_to_canonical_oracle(self):
        days, frame, params, period = fixture(count=170, start='2020-09-01')
        frame = frame.drop(index=pd.Timestamp('2021-01-01'), errors='ignore')
        days = pd.DatetimeIndex(frame.index)
        boundary = pd.Timestamp('2021-01-01')
        past = replace(period, end=boundary - pd.Timedelta(days=1))
        present = replace(period, start=boundary, end=None)
        positions = np.flatnonzero((days >= boundary-pd.Timedelta(days=1)) &
                                   (days < boundary+pd.Timedelta(days=35)))
        original = compute_counterfactual_trade40_batch(
            frame, positions, ticker='2330', schedule=(present,), exchange_calendar=days)
        with_teacher = compute_counterfactual_trade40_batch(
            frame, positions, ticker='2330', schedule=(past, present),
            exchange_calendar=days, signal_state_reset_date=boundary)
        np.testing.assert_array_equal(with_teacher.valid_mask, original.valid_mask)
        np.testing.assert_array_equal(with_teacher.target_raw_r, original.target_raw_r)
        np.testing.assert_array_equal(with_teacher.maturity_dates, original.maturity_dates)

    def test_weekend_fit_cutoff_censors_forced_closeout_but_research_end_accepts(self):
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITSource, materialize_formal_mintrade40_pit_labels,
        )
        days, frame, params, period = fixture(count=140, start='2018-01-02')
        source = FormalMinTrade40PITSource(
            seed=42, param_sha256='a'*64, schedule=(period,),
            publication_dates=('2018-01-01',))
        for date in ('2018-05-25', '2018-05-27'):
            kwargs = dict(source=source, expected_seed=42, frame=frame,
                positions=np.array([90]), ticker='2330', exchange_calendar=days,
                fit_cutoff=date)
            fit = materialize_formal_mintrade40_pit_labels(**kwargs)
            study = materialize_formal_mintrade40_pit_labels(
                **kwargs, research_end_is_terminal=True)
            self.assertFalse(fit.completed[0])
            self.assertTrue(np.isnan(fit.raw_net_r[0]))
            self.assertTrue(study.completed[0])
            canonical = compute_counterfactual_trade40_batch(
                frame.loc[:date], np.array([90]), ticker='2330',
                schedule=(period,), exchange_calendar=days[days <= date],
                research_cutoff=date)
            self.assertAlmostEqual(float(study.raw_net_r[0]), float(canonical.target_raw_r[0]))
            self.assertTrue(bool(canonical.research_cutoff_forced_mask[0]))

    def test_teacher_asof_fit_identity_excludes_future_but_binds_current_history(self):
        from services.breakout_quality.fitting_data_dependencies import FitDataDependencies
        from core.mintrade40_teacher_asof import mintrade40_teacher_asof_sha256
        from services.breakout_quality.formal_mintrade40_pit_source import (
            FormalMinTrade40PITSource, slice_formal_mintrade40_pit_source_for_fold,
        )
        days, frame, params, period = fixture(count=170, start='2020-09-01')
        future_start = days[150]
        past = replace(period, end=future_start-pd.Timedelta(days=1))
        future = replace(period, start=future_start, end=None)
        published = str(days[0].date())
        late_published = str((future_start-pd.Timedelta(days=1)).date())
        payload = lambda p,d: {'start':str(p.start.date()),
            'end':None if p.end is None else str(p.end.date()),
            'params_signature':p.params_signature, 'published':d}
        parent = FormalMinTrade40PITSource(seed=42, param_sha256='a'*64,
            schedule=(past, future), publication_dates=(published, late_published))
        _, original_digest = slice_formal_mintrade40_pit_source_for_fold(
            parent, fit_cutoff=days[120])
        truth = {'target_kind':'formal_seed_mintrade40_broker_net_r',
                 'teacher_lineage':{'contract':'formal_mintrade40_seed_historical_plus_p16_v1',
                    'seed':42, 'oos_start_date':None}}
        bare = {'target_contract':truth,
                'formal_mintrade40_teacher_periods':[payload(past,published)]}
        extended = {'target_contract':truth,
                    'formal_mintrade40_teacher_periods':[payload(past,published),
                                                         payload(future,late_published)]}
        first = FitDataDependencies({},bare).rolling_source(fit_cutoff=days[120])
        later = FitDataDependencies({},extended).rolling_source(fit_cutoff=days[120])
        self.assertEqual(first,later)
        self.assertEqual(first['target_contract']['teacher_lineage']['fold_teacher_as_of_sha256'],
                         original_digest)
        mutated = {**bare, 'formal_mintrade40_teacher_periods':[
            {**payload(past,published), 'params_signature':'different'}]}
        self.assertNotEqual(first,FitDataDependencies({},mutated).rolling_source(
            fit_cutoff=days[120]))
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            FitDataDependencies({},bare).rolling_source()

    def test_fit_source_binds_actual_ordered_training_truth(self):
        from services.breakout_quality.fitting_data_dependencies import mintrade40_fit_sample_sha256
        from types import SimpleNamespace
        frame = pd.DataFrame({'ticker':['2330','2317'],
            'date':pd.to_datetime(['2020-01-02','2020-01-02']),
            'source_pos':[99,98],
            'label_eval_end_date':pd.to_datetime(['2020-02-28','2020-02-27'])})
        bundle = SimpleNamespace(target_manifest={'target_contract':{
            'target_kind':'formal_seed_mintrade40_broker_net_r'}},
            group_table=frame, raw_target=np.array([0.5,-0.2],dtype=np.float32))
        original = mintrade40_fit_sample_sha256(bundle, np.array([0,1]))
        self.assertNotEqual(original, mintrade40_fit_sample_sha256(bundle,np.array([1,0])))
        bundle.raw_target[1]=0.1
        self.assertNotEqual(original, mintrade40_fit_sample_sha256(bundle,np.array([0,1])))

    def test_forward_rolling_fit_bridge_requires_original_labeled_fit_witness(self):
        from services.breakout_quality.fitting_data_dependencies import (
            FitDataDependencies, forward_publication_fit_dependency_issues,
        )
        from core.breakout_quality_registry import (
            DAILY_UNIVERSAL_FORMAL_MINTRADE40_NETR_PAIRWISE_PROFILE,
        )
        published_target = {
            'schema_version': 1,
            'target_id': 'formal_mintrade40',
            'target_contract': {
                'target_kind': 'formal_seed_mintrade40_broker_net_r',
                'teacher_lineage': {
                    'contract': 'formal_mintrade40_seed_historical_plus_p16_v1',
                    'seed': 42, 'oos_start_date': '2021-01-01',
                },
            },
            'formal_mintrade40_teacher_periods': [
                {'start': '2019-01-01', 'end': '2020-12-31',
                 'params_signature': 'historical', 'published': '2018-12-31'},
                {'start': '2021-01-01', 'end': None,
                 'params_signature': 'p16', 'published': '2020-12-31'},
            ],
        }
        published_dataset = {
            'dataset': 'full', 'policy': {'repro': True},
            'source_data_inventory': {'inventory': 'verified'},
            'dataset_artifacts': {'events': 'verified'},
        }
        cutoff = '2020-06-01'
        expected = FitDataDependencies(
            published_dataset, published_target,
        ).rolling_source(fit_cutoff=cutoff)
        original_samples = 'a' * 64
        expected['formal_mintrade40_fit_samples_sha256'] = original_samples
        kwargs = {
            'source_dataset': published_dataset,
            'source_target': published_target,
            'expected_fold_source': expected,
            'current_target': published_target,
            'experiment_profile': DAILY_UNIVERSAL_FORMAL_MINTRADE40_NETR_PAIRWISE_PROFILE,
            'project_root': Path.cwd(), 'fit_cutoff': cutoff,
            'inventory_matcher': lambda *, published, current, project_root: published == current,
        }
        witness = {
            'target_contract': expected['target_contract'],
            'formal_mintrade40_fit_samples_sha256': original_samples,
        }
        self.assertEqual((), forward_publication_fit_dependency_issues(
            **kwargs, published_forward_fit_source=witness,
        ))
        self.assertTrue(any('fit_samples' in error for error in
            forward_publication_fit_dependency_issues(
                **kwargs, published_forward_fit_source={
                    **witness, 'formal_mintrade40_fit_samples_sha256': 'b' * 64,
                },
            )))
        self.assertIn('witness missing', ' '.join(
            forward_publication_fit_dependency_issues(
                **kwargs, published_forward_fit_source=None,
            )))
        tampered = {**witness, 'target_contract': {'target_kind': 'changed'}}
        self.assertIn('teacher witness mismatch', ' '.join(
            forward_publication_fit_dependency_issues(
                **kwargs, published_forward_fit_source=tampered,
            )))

    def test_standard_model_sop_uses_independent_40d_price_maturity(self):
        from services.breakout_quality.standard_model_sop import build_standard_model_sop
        days = pd.to_datetime(['2021-01-04','2021-01-04','2021-01-05','2021-01-05'])
        table = pd.DataFrame({
            'ticker':['2330','2317','2330','2317'], 'date':days,
            'group_index':[0,1,2,3], 'source_pos':[100,100,140,140],
            'label':[1,0,1,0],
            'standard_common_40d_matured':[True,True,False,False],
            'target_favorable_r':[2.,1.,np.nan,np.nan],
            'target_adverse_r':[1.,0.5,np.nan,np.nan],
            'target_favorable_return':[.08,.04,np.nan,np.nan],
            'target_adverse_return_to_peak':[.03,.02,np.nan,np.nan],
        })
        scores=pd.DataFrame({'group_index':[0,1,2,3],'date':days,
                    'breakout_quality_score':[.4,.7,.3,.5]})
        report=build_standard_model_sop(group_table=table,
            raw_target=np.array([.5,.9,.2,.7],dtype=np.float32),
            training_objective='daily_pairwise_ranking',
            validation_scores=scores, oos_scores=scores,
            breakout_scores=scores, evaluation_mode='rolling_oos',
            evaluation_horizon_bars=40)
        self.assertEqual(report['schema'],'standard_model_sop_v10')
        # Missing *mature* price truth must still block the report.
        table.loc[0,'target_favorable_r']=np.nan
        with self.assertRaisesRegex(ValueError,'actual truth'):
            build_standard_model_sop(group_table=table,
                raw_target=np.array([.5,.9,.2,.7],dtype=np.float32),
                training_objective='daily_pairwise_ranking',
                validation_scores=scores, oos_scores=scores,
                breakout_scores=scores, evaluation_mode='rolling_oos',
                evaluation_horizon_bars=40)


class RealPathCommonPriceEvaluationCases(unittest.TestCase):
    """Scheme B: fixed price cohorts, target-null stays unknown for FIT."""

    @staticmethod
    def _evaluated(model_target_is_sparse: bool):
        from services.breakout_quality.standard_model_sop import build_standard_model_sop
        n = 60
        days = pd.to_datetime(['2024-01-02'] * 30 + ['2024-01-03'] * 30)
        table = pd.DataFrame({
            'ticker': [f'{1000 + i%30:04}' for i in range(n)],
            'date': days, 'group_index': np.arange(n),
            'source_pos': np.full(n, 80), 'label': np.ones(n),
            'target_favorable_r': np.linspace(.3, 2, n),
            'target_adverse_r': np.linspace(.1, .9, n),
            'target_favorable_return': np.linspace(.02, .2, n),
            'target_adverse_return_to_peak': np.linspace(.01, .07, n),
        })
        target = np.linspace(-.8, 1.5, n).astype('float32')
        if model_target_is_sparse:
            table['standard_common_40d_matured'] = np.ones(n, dtype=bool)
            target[[0, 9, 29, 32, 41, 59]] = np.nan
        score = pd.DataFrame({
            'group_index': np.arange(n), 'date': days,
            'breakout_quality_score': np.linspace(.01, .99, n),
        })
        report = build_standard_model_sop(
            group_table=table, raw_target=target,
            training_objective='daily_pairwise_ranking',
            validation_scores=score, oos_scores=score, breakout_scores=score,
            evaluation_mode='rolling_oos', evaluation_horizon_bars=40,
        )
        return table, target, score, report

    def test_common_40d_cohorts_include_null_min_labels_without_fabricating_training_truth(self):
        normal_table, normal_target, _, normal = self._evaluated(False)
        trade_table, trade_target, _, trade = self._evaluated(True)
        self.assertEqual(
            normal['evaluation_truth']['cohort_identity'],
            trade['evaluation_truth']['cohort_identity'],
        )
        for split in ('validation', 'oos', 'breakout_candidate_oos'):
            self.assertEqual(normal['upside_downside_alignment_evaluation'][split]['group_count'], 60)
            self.assertEqual(trade['upside_downside_alignment_evaluation'][split]['group_count'], 60)
            self.assertEqual(trade['split_metrics'][split]['group_count'], 54)
            self.assertEqual(normal['split_metrics'][split]['group_count'], 60)
            self.assertEqual(
                trade['upside_downside_alignment_evaluation'][split]['score_to_full_mfe_daily_spearman'],
                normal['upside_downside_alignment_evaluation'][split]['score_to_full_mfe_daily_spearman'],
            )
        self.assertEqual(int(np.isnan(trade_target).sum()), 6)
        self.assertEqual(int(np.isfinite(normal_target).sum()), 60)
        self.assertTrue(trade_table['standard_common_40d_matured'].all())
        self.assertNotIn('standard_common_40d_matured',normal_table.columns)

    def test_cross_model_compare_accepts_same_price_population_with_different_supervised_targets(self):
        from services.research.breakout_quality_application import _standard_model_comparison_views
        _t1, _r1, _f1, control = self._evaluated(False)
        _t2, _r2, _f2, actual_path = self._evaluated(True)
        models = [
            {'model_id': 'PRICE', 'payload': {'standard_model_sop': control}},
            {'model_id': 'REAL_PATH', 'payload': {'standard_model_sop': actual_path}},
        ]
        self.assertEqual(len(_standard_model_comparison_views(models)), 2)
        self.assertEqual(
            control['evaluation_truth']['price_truth_identity'],
            actual_path['evaluation_truth']['price_truth_identity'],
        )

    def test_price_population_sha_alone_does_not_mask_different_actual_price_truth(self):
        from services.research.breakout_quality_application import _standard_model_comparison_views
        from services.breakout_quality.standard_model_sop import build_standard_model_sop
        table, target, score, control = self._evaluated(False)
        changed = table.copy()
        changed.loc[5, 'target_favorable_return'] += 0.001
        other = build_standard_model_sop(
            group_table=changed, raw_target=target,
            training_objective='daily_pairwise_ranking',
            validation_scores=score, oos_scores=score, breakout_scores=score,
            evaluation_mode='rolling_oos', evaluation_horizon_bars=40,
        )
        self.assertEqual(control['evaluation_truth']['cohort_identity'],
                         other['evaluation_truth']['cohort_identity'])
        self.assertNotEqual(control['evaluation_truth']['price_truth_identity'],
                            other['evaluation_truth']['price_truth_identity'])
        with self.assertRaisesRegex(ValueError, 'price truth SHA differs'):
            _standard_model_comparison_views([
                {'model_id':'PRICE','payload':{'standard_model_sop':control}},
                {'model_id':'ALTERED','payload':{'standard_model_sop':other}},
            ])

    def test_existing_price_target_report_unmodified_when_new_source_is_added(self):
        from services.breakout_quality.standard_model_sop import build_standard_model_sop
        table, target, score, old = self._evaluated(False)
        fresh = table.copy()
        fresh['standard_common_40d_matured'] = np.ones(len(fresh), dtype=bool)
        now = build_standard_model_sop(
            group_table=fresh, raw_target=target,
            training_objective='daily_pairwise_ranking',
            validation_scores=score, oos_scores=score, breakout_scores=score,
            evaluation_mode='rolling_oos', evaluation_horizon_bars=40,
        )
        self.assertEqual(old, now)

    def test_null_rows_with_missing_price_truth_must_block_not_fill_with_zero(self):
        from services.breakout_quality.standard_model_sop import build_standard_model_sop
        table, target, score, _ = self._evaluated(True)
        table.loc[9, 'target_favorable_return'] = np.nan
        with self.assertRaisesRegex(ValueError, 'complete price evidence'):
            build_standard_model_sop(
                group_table=table, raw_target=target,
                training_objective='daily_pairwise_ranking',
                validation_scores=score, oos_scores=score, breakout_scores=score,
                evaluation_mode='rolling_oos', evaluation_horizon_bars=40,
            )

    def test_fold_train_and_epoch_validation_stay_target_gated_but_price_scoring_does_not(self):
        from filters.breakout_quality.pit_group_selection import _fold_group_ids
        from filters.breakout_quality.pit_group_selection import TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS
        dates = pd.to_datetime([
            '2023-01-02','2023-01-03',  # historical FIT
            '2023-12-06','2023-12-06','2023-12-07', # validation
            '2024-01-02','2024-01-03', # OOS score
        ])
        table = pd.DataFrame({
            'date': dates,
            'label_eval_end_date': pd.to_datetime([
                '2023-03-05','2023-03-06','2023-12-20',None,'2023-12-21',None,None
            ]),
            'standard_common_40d_price_end_date': pd.to_datetime([
                '2023-03-05','2023-03-06','2023-12-20','2023-12-22','2023-12-21',None,None
            ]),
        })
        bundle=SimpleNamespace(group_table=table,
            target_valid=np.asarray([True,True,True,False,True,False,False]),
            profile=SimpleNamespace(training_sample_scope=TRAINING_SAMPLE_SCOPE_DAILY_ELIGIBLE_STOCK_DAYS))
        fold={'fold_id':'toy','score_start':pd.Timestamp('2024-01-01'),
              'score_end':pd.Timestamp('2024-02-29')}
        ids=_fold_group_ids(bundle,fold,validation_months=1)
        self.assertEqual(ids['train_ids'].tolist(),[0,1])
        self.assertEqual(ids['validation_ids'].tolist(),[2,4])
        self.assertEqual(ids['final_ids'].tolist(),[0,1,2,4])
        self.assertEqual(ids['standard_validation_ids'].tolist(),[2,3,4])
        self.assertEqual(ids['score_ids'].tolist(),[5,6])
        from services.breakout_quality.point_in_time_scores import _standard_validation_score_ids
        self.assertEqual(_standard_validation_score_ids(ids).tolist(),[2,3,4])

    def test_signed_validation_rows_must_match_full_price_cohort_exactly(self):
        from filters.breakout_quality.pit_group_selection import (
            validate_standard_validation_sidecar_membership,
        )
        expected = np.asarray([0, 2, 3, 5], dtype=np.int64)
        validate_standard_validation_sidecar_membership(
            expected, expected.copy(), fold_id="synthetic_fold",
        )
        with self.assertRaisesRegex(ValueError, "missing=1, extra=0"):
            validate_standard_validation_sidecar_membership(
                expected, np.asarray([0, 2, 5]), fold_id="synthetic_fold",
            )
        with self.assertRaisesRegex(ValueError, "missing=0, extra=1"):
            validate_standard_validation_sidecar_membership(
                expected, np.asarray([0, 1, 2, 3, 5]), fold_id="synthetic_fold",
            )
        with self.assertRaisesRegex(ValueError, "validation price cohort differs"):
            validate_standard_validation_sidecar_membership(
                expected, np.asarray([5, 3, 2, 0]), fold_id="synthetic_fold",
            )
        with self.assertRaisesRegex(ValueError, "validation price cohort differs"):
            validate_standard_validation_sidecar_membership(
                expected, np.asarray([0, 2, 3, 3]), fold_id="synthetic_fold",
            )

    def test_published_pit_aggregate_preserves_real_path_validation_scope(self):
        """A repaired Fold must remain READY after the aggregate is republished."""
        from services.breakout_quality.point_in_time_scores import _combined_fold_record
        from filters.breakout_quality.pit_group_selection import (
            STANDARD_PRICE_VALIDATION_SCORE_SCOPE,
        )
        from services.breakout_quality.point_in_time_scores import FOLD_MANIFEST_FILENAME

        with TemporaryDirectory() as tmp:
            args = SimpleNamespace(point_in_time_dir_override=tmp)
            fold_id = "fold_20210104_20211231"
            fold_dir = Path(tmp) / "folds" / fold_id
            fold_dir.mkdir(parents=True)
            (fold_dir / FOLD_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
            source = {
                "fold_id": fold_id,
                "planned_periods": {"validation_start": "2020-01-01", "score_start": "2021-01-04"},
                "observed_periods": {},
                "model_information_cutoff": "2021-01-03",
                "group_counts": {},
                "event_row_counts": {},
                "selected_epoch": 3,
                "contract_fingerprint": "synthetic",
                "artifacts": {},
                "validation_score_scope": STANDARD_PRICE_VALIDATION_SCORE_SCOPE,
            }
            scored = _combined_fold_record(args, source)
            self.assertEqual(
                scored["validation_score_scope"], STANDARD_PRICE_VALIDATION_SCORE_SCOPE,
            )
            source.pop("validation_score_scope")
            legacy = _combined_fold_record(args, source)
            self.assertNotIn("validation_score_scope", legacy)

    def test_stale_validation_sidecar_scope_must_not_claim_ready_score_source(self):
        """Regression for MR-13CM: old signed OOS SCORE != new SOP Validation."""
        import json
        from hashlib import sha256
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.breakout_quality_registry import get_breakout_quality_experiment_profile
        from core.breakout_quality_runtime import get_continuous_ranker_score_output_columns
        from core.breakout_quality_runtime_resolver import get_continuous_ranker_execution_recipe
        from core.breakout_quality_runtime import CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH
        from filters.breakout_quality.ranker_sample_contract import build_score_eligibility_contract
        from filters.breakout_quality.pit_group_selection import (
            STANDARD_PRICE_VALIDATION_SCORE_SCOPE,
            required_standard_validation_score_scope,
        )
        from filters.breakout_quality import ranking_score_store as score_owner
        from filters.breakout_quality.paths import (
            SELECTION_POINT_IN_TIME_SCORE_FILENAME,
            SELECTION_POINT_IN_TIME_MANIFEST_FILENAME,
            SELECTION_POINT_IN_TIME_COVERAGE_FILENAME,
        )
        from core.breakout_quality_policy import get_breakout_quality_model_test_settings

        entries = get_breakout_quality_model_test_settings().model_profiles
        real_profiles = [profile for _name, profile in entries
                         if get_continuous_ranker_execution_recipe(profile).target_policy.materialization_mode
                         == CONTINUOUS_RANKER_TARGET_MATERIALIZATION_DAILY_TRADE_PATH]
        price_profiles = [profile for _name, profile in entries
                          if required_standard_validation_score_scope(profile) is None]
        self.assertTrue(real_profiles)
        self.assertTrue(price_profiles)

        def verify_source(profile_name, declared_scope):
            settings = get_breakout_quality_workflow_settings(experiment_profile=profile_name)
            profile = get_breakout_quality_experiment_profile(profile_name)
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                score = root / SELECTION_POINT_IN_TIME_SCORE_FILENAME
                coverage = root / SELECTION_POINT_IN_TIME_COVERAGE_FILENAME
                manifest_path = root / SELECTION_POINT_IN_TIME_MANIFEST_FILENAME
                score.write_text('score-placeholder', encoding='utf-8')
                coverage.write_text('coverage-placeholder', encoding='utf-8')

                def record(path):
                    data = path.read_bytes()
                    return {"filename": path.name, "sha256": sha256(data).hexdigest(),
                            "size_bytes": len(data)}

                fold = {"fold_id": "fold_20210104_20211231"}
                if declared_scope is not None:
                    fold["validation_score_scope"] = declared_scope
                manifest = {
                    "filter_id": settings.filter_id,
                    "model_architecture": settings.model_architecture,
                    "experiment_profile": profile_name,
                    "training_sample_scope": profile.training_sample_scope,
                    "seed": int(settings.seed), "status": "BUILT",
                    "continuous_target_id": profile.continuous_target_id,
                    "source_continuous_target": {"target_contract": {"schema": "fixture-v1"}},
                    "score_period": {"start": "2021-01-04", "end": "2021-12-31"},
                    "folds": [fold], "artifacts": {"scores": record(score), "coverage": record(coverage)},
                    "score_columns": get_continuous_ranker_score_output_columns(profile.training_objective),
                    "score_eligibility_contract": build_score_eligibility_contract(profile),
                    "lookahead_contract": score_owner.selection_pit_lookahead_contract(),
                }
                manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
                kwargs = {
                    "project_root": root,
                    "filter_id": settings.filter_id,
                    "model_architecture": settings.model_architecture,
                    "experiment_profile": profile_name,
                    "point_in_time_dir_override": root,
                    "allow_stale_source": True,
                }
                return score_owner.load_selection_point_in_time_score_source_contract(**kwargs)

        with self.assertRaisesRegex(ValueError, "validation score scope stale/missing"):
            verify_source(real_profiles[0], None)
        with self.assertRaisesRegex(ValueError, "validation score scope stale/missing"):
            verify_source(real_profiles[0], "supervised_target_only")
        ready = verify_source(real_profiles[0], STANDARD_PRICE_VALIDATION_SCORE_SCOPE)
        self.assertEqual(ready.experiment_profile, real_profiles[0])
        old = verify_source(price_profiles[0], None)
        self.assertEqual(old.experiment_profile, price_profiles[0])

    def test_missing_score_only_source_does_not_authorize_refit(self):
        from services.breakout_quality.point_in_time_scores import _standard_validation_score_ids
        ids={'validation_ids':np.asarray([0,2],dtype=np.int64)}
        self.assertEqual(_standard_validation_score_ids(ids).tolist(),[0,2])
        # The existing FIT validator remains authoritative; the expanded
        # score-only cohort must never enter epoch selection or fit_final.



class FormalMinTrade40UpstreamPlanRegressionCases(unittest.TestCase):
    """Regression: model work cannot silently run a historical optimizer."""

    def test_read_only_missing_history_never_promotes_legacy_p2(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            inspect_formal_mintrade40_optimizer_teacher,
        )
        seed = 693545351
        with TemporaryDirectory() as temp:
            root = Path(temp)
            legacy = root / 'outputs' / 'optimizer' / 'historical_p2' / 'active_params'
            legacy.mkdir(parents=True)
            (legacy / 'roos_base_best.json').write_text('{}', encoding='utf-8')
            before = {p.relative_to(root) for p in root.rglob('*') if p.is_file()}
            state = inspect_formal_mintrade40_optimizer_teacher(project_root=root, seed=seed)
            after = {p.relative_to(root) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(state['action'], 'BUILD')
            self.assertIn('mintrade40_label_history', str(state['path']))
            self.assertEqual(before, after)

    def test_existing_corrupt_historical_publication_blocks_without_overwriting(self):
        from services.optimizer.formal_mintrade40_teacher_source import (
            inspect_formal_mintrade40_optimizer_teacher,
            formal_mintrade40_history_params_path,
        )
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = formal_mintrade40_history_params_path(root, 693545351)
            target.parent.mkdir(parents=True)
            target.write_text('{}', encoding='utf-8')
            state = inspect_formal_mintrade40_optimizer_teacher(
                project_root=root, seed=693545351,
            )
            self.assertEqual(state['action'], 'BLOCKED')
            self.assertEqual(target.read_text(encoding='utf-8'), '{}')

    def test_partial_history_exact_preflight_resumes_otherwise_blocks(self):
        from core.file_integrity import canonical_json_sha256
        from services.optimizer import formal_mintrade40_teacher_source as teacher
        seed = 693545351
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = teacher.formal_mintrade40_history_params_path(root, seed)
            target.parent.parent.mkdir(parents=True)
            preflight_path = target.parent.parent / 'rolling_preflight.json'
            payload = {'arm_id': teacher.HISTORICAL_MINTRADE40_TRAINING_SET,
                       'dataset': 'full', 'trials_per_fold': 300}
            digest = canonical_json_sha256(payload)
            with patch.object(teacher, '_expected_historical_runtime_identity', return_value=digest):
                import json
                preflight_path.write_text(json.dumps({**payload,
                    'runtime_identity_sha256': digest, 'status': 'RUNNING'}), encoding='utf-8')
                self.assertEqual(teacher.inspect_formal_mintrade40_optimizer_teacher(
                    project_root=root, seed=seed)['action'], 'RESUME')
                preflight_path.write_text(json.dumps({**payload,
                    'dataset': 'different', 'runtime_identity_sha256': digest}), encoding='utf-8')
                self.assertEqual(teacher.inspect_formal_mintrade40_optimizer_teacher(
                    project_root=root, seed=seed)['action'], 'BLOCKED')
            with patch.object(teacher, '_expected_historical_runtime_identity', return_value='b'*64):
                self.assertEqual(teacher.inspect_formal_mintrade40_optimizer_teacher(
                    project_root=root, seed=seed)['action'], 'BLOCKED')

    def test_historical_preflight_fingerprint_inspection_never_writes_dataset_witness(self):
        from services.optimizer import formal_mintrade40_teacher_source as teacher
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch('filters.breakout_quality.source_inventory._build_source_data_inventory',
                       return_value={'schema_version': 1, 'dataset_profile': 'full',
                                     'csv_inventory_sha256': 'a'*64}), \
                 patch('filters.breakout_quality.source_inventory.build_source_data_inventory',
                       side_effect=AssertionError('Read-only preflight cannot persist a content witness')):
                digest = teacher._expected_historical_runtime_identity(root)
            self.assertEqual(len(digest), 64)
            self.assertFalse((root / 'outputs').exists())

    def test_completed_history_reuses_only_current_original_runtime(self):
        from services.optimizer import formal_mintrade40_teacher_source as teacher
        seed = 693545351
        with TemporaryDirectory() as temp:
            path, _payload, preflight, _summary = FormalMinTrade40OptimizerAttestationCases._artifacts(
                temp, seed=seed,
            )
            with patch.object(teacher, '_expected_historical_runtime_identity',
                              return_value=preflight['runtime_identity_sha256']):
                self.assertEqual(teacher.inspect_formal_mintrade40_optimizer_teacher(
                    project_root=temp, seed=seed)['action'], 'REUSE')
            with patch.object(teacher, '_expected_historical_runtime_identity', return_value='b'*64):
                self.assertEqual(teacher.inspect_formal_mintrade40_optimizer_teacher(
                    project_root=temp, seed=seed)['action'], 'BLOCKED')
            self.assertTrue(path.is_file())

    def test_retired_trade_path_profile_cannot_claim_formal_teacher_optimizer(self):
        from services.research import breakout_quality_application as app
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        settings = get_breakout_quality_workflow_settings(experiment_profile=OLD_PROFILE)
        base = ResearchArtifactPlan.from_actions([ResearchArtifactAction(
            action_id='model-upstream:dataset_core', artifact_key='model-upstream:dataset_core',
            action='REUSE', builder_type=None, description='READY', path='dataset_summary.json',
        )])
        with patch.object(app, 'collect_model_upstream_preparation_plan', return_value=base), \
             patch('services.optimizer.formal_mintrade40_teacher_source.inspect_formal_mintrade40_optimizer_teacher',
                   side_effect=AssertionError('Retired target cannot trigger formal Min preflight')):
            plan = app._collect_continuous_research_input_plan(settings)
        self.assertEqual(plan, base)

    def test_model_preplan_exposes_two_original_seed_dependent_sources(self):
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        from services.research import breakout_quality_application as app
        from services.optimizer import formal_mintrade40_teacher_source as history
        from services.breakout_quality import formal_mintrade40_training_source as current
        settings = get_breakout_quality_workflow_settings(
            experiment_profile='daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        )
        base = ResearchArtifactPlan.from_actions([ResearchArtifactAction(
            action_id='model-upstream:dataset_core', artifact_key='model-upstream:dataset_core',
            action='REUSE', builder_type=None, description='READY', path='dataset_summary.json',
            execution_priority=10,
        )])
        seed = 693545351
        def status(*, project_root, seed):
            return {'action': 'BUILD', 'path': Path(project_root) / 'seed' / str(seed),
                    'reason': 'Genuine optimizer needed'}
        with patch.object(app, 'collect_model_upstream_preparation_plan', return_value=base), \
             patch.object(history, 'inspect_formal_mintrade40_optimizer_teacher', side_effect=status) as h, \
             patch.object(current, 'inspect_formal_mintrade40_p16_teacher', side_effect=status) as p:
            plan = app._collect_continuous_research_input_plan(settings, seed=seed)
        self.assertEqual(plan.overall_status, 'PREPARABLE')
        self.assertEqual(len(plan.actions), 3)
        self.assertEqual([row.artifact_key for row in plan.actions], [
            'model-upstream:dataset_core',
            f'model-upstream:formal_mintrade40_p16_seed_{seed}',
            f'model-upstream:formal_mintrade40_history_seed_{seed}',
        ])
        self.assertEqual([row.action for row in plan.actions], ['REUSE', 'BUILD', 'BUILD'])
        self.assertTrue(all(row.dependencies == ('model-upstream:dataset_core',)
                            for row in plan.actions[1:]))
        h.assert_called_once_with(project_root=app.PROJECT_ROOT, seed=seed)
        p.assert_called_once_with(project_root=app.PROJECT_ROOT, seed=seed)

    def test_optimizer_build_is_only_permitted_through_model_upstream_plan(self):
        from services.research import breakout_quality_application as app
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        from services.breakout_quality import formal_mintrade40_training_source as owner
        from services.optimizer import formal_mintrade40_teacher_source as history
        settings = get_breakout_quality_workflow_settings(
            experiment_profile='daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        )
        seed = 693545351
        actions = [ResearchArtifactAction(
            action_id='model-upstream:dataset_core', artifact_key='model-upstream:dataset_core',
            action='REUSE', builder_type=None, description='READY', path='dataset_summary.json',
        )]
        def fake_plan():
            return ResearchArtifactPlan.from_actions(actions)
        def fake_inspector(*, project_root, seed):
            return {'action':'BUILD','path':Path(project_root)/'seed' / str(seed),
                    'reason':'Missing original'}
        with patch.object(app, 'collect_model_upstream_preparation_plan', side_effect=lambda *a, **kw: fake_plan()), \
             patch.object(history, 'inspect_formal_mintrade40_optimizer_teacher', side_effect=fake_inspector), \
             patch.object(owner, 'inspect_formal_mintrade40_p16_teacher', side_effect=fake_inspector), \
             patch.object(owner, '_current_p16_seed_artifact', side_effect=RuntimeError('stop')) as builder:
            # Even if missing both, planning may not call the expensive producer.
            plan = app._collect_continuous_research_input_plan(settings, seed=seed)
            self.assertEqual(plan.overall_status, 'PREPARABLE')
            builder.assert_not_called()

    def test_robustness_seed_preparation_never_falls_back_to_default_seed(self):
        from services.research import breakout_quality_application as app
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        from core.research_orchestration import ResearchArtifactAction, ResearchArtifactPlan
        settings = get_breakout_quality_workflow_settings(
            experiment_profile='daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        )
        seed = 2014432738
        plan = ResearchArtifactPlan.from_actions([ResearchArtifactAction(
            action_id='model-upstream:dataset_core', artifact_key='model-upstream:dataset_core',
            action='BUILD', builder_type='dataset', description='Missing', path='dataset',
        )])
        with patch.object(app, '_collect_continuous_research_input_plan', return_value=plan) as inspect, \
             patch.object(app, '_render_continuous_research_input_plan') as display, \
             patch.object(app, '_prepare_continuous_research_inputs', return_value=0) as prepare:
            self.assertEqual(app._prepare_formal_robustness_seed_inputs('research', settings, seed=seed), 0)
        inspect.assert_called_once_with(settings, seed=seed)
        prepare.assert_called_once_with('research', settings, seed=seed)
        display.assert_called_once_with(settings, plan)

    def test_rolling_audit_only_must_not_prepare_min_or_refit(self):
        from services.research import breakout_quality_application as app
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        settings = get_breakout_quality_workflow_settings(
            experiment_profile='daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        )
        with patch.object(app, '_rolling_robustness_score_core_ready', return_value=(True, None)), \
             patch.object(app, '_prepare_formal_robustness_seed_inputs') as forbidden, \
             patch.object(app, '_run_command', return_value=0) as run:
            self.assertEqual(app._build_rolling_robustness_seed('research', settings,
                                                                seed=2014432738), 0)
        forbidden.assert_not_called()
        self.assertEqual(run.call_args.args[0], 'audit-point-in-time-scores')

    def test_rolling_missing_score_prepares_exact_seed_even_if_dataset_prepared(self):
        from services.research import breakout_quality_application as app
        from core.breakout_quality_policy import get_breakout_quality_workflow_settings
        settings = get_breakout_quality_workflow_settings(
            experiment_profile='daily_universal_formal_mintrade40_netr_per_seed_full_list_ndcg_pairwise'
        )
        seed = 2014432738
        with patch.object(app, '_rolling_robustness_score_core_ready', return_value=(False, 'Missing')), \
             patch.object(app, '_prepare_formal_robustness_seed_inputs', return_value=2) as prepare, \
             patch.object(app, '_run_command', side_effect=AssertionError('Cannot run')):
            self.assertEqual(app._build_rolling_robustness_seed(
                'research', settings, seed=seed, inputs_prepared=True,
            ), 2)
        prepare.assert_called_once_with('research', settings, seed=seed)

    def test_daily_materializers_must_not_auto_authorize_optimizer(self):
        import inspect
        from services.breakout_quality import point_in_time_scores, train_daily_ranker
        self.assertIn('authorized_target_dependency_build=False',
                      inspect.getsource(point_in_time_scores))
        self.assertIn('authorized_target_dependency_build=False',
                      inspect.getsource(train_daily_ranker))
        self.assertNotIn('authorized_target_dependency_build=True',
                         inspect.getsource(point_in_time_scores))
        self.assertNotIn('authorized_target_dependency_build=True',
                         inspect.getsource(train_daily_ranker))


if __name__ == "__main__":
    unittest.main()
