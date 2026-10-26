"""Canonical Min ROOS hypothetical-40-session execution primitive regressions.

The retired MR-13CL P2/P16 teacher is deliberately not an executable path.
No formal Windows suite, fitting, optimizer, or real-market Replay is run here.
"""
from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
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
        # A valid 40D label must not be copied to an immature 60D sample.
        forty_late = compute_counterfactual_trade40_batch(
            frame, np.array([90]), ticker="2330", schedule=schedule,
            exchange_calendar=days,
        )
        sixty_late = compute_counterfactual_trade40_batch(
            frame, np.array([90]), ticker="2330", schedule=schedule,
            exchange_calendar=days, horizon_sessions=60,
        )
        self.assertFalse(sixty_late.valid_mask[0])
        self.assertEqual(sixty_late.reasons[0], "unmatured_or_missing_exchange_session")
        self.assertTrue(forty_late.valid_mask[0])
        with self.assertRaisesRegex(ValueError, "positive exchange sessions"):
            compute_counterfactual_trade40_batch(
                frame, np.array([10]), ticker="2330", schedule=schedule,
                exchange_calendar=days, horizon_sessions=0,
            )

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
        self.assertEqual(batch.reasons[0], "unmatured_or_missing_exchange_session")

    def test_no_prior_parameter_is_not_backfilled(self):
        days, frame, params, period = fixture()
        formal = (ExecutableMinPeriod(days[50], None, params, "formal_min_roos_seed", "s"),)
        batch = compute_counterfactual_trade40_batch(
            frame, np.asarray([48]), ticker="2330",
            schedule=formal, exchange_calendar=days,
        )
        self.assertFalse(batch.valid_mask[0])
        self.assertEqual(batch.reasons[0], "no_historical_effective_Min_param")


if __name__ == "__main__":
    unittest.main()
