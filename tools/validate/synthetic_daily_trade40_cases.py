"""Canonical Min ROOS hypothetical-40-session execution primitive regressions.

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
                    self.assertEqual(actual.reasons, ("unfilled_superseded_by_new_setup",))
                    self.assertEqual(str(actual.maturity_dates[0]), "2026-01-02")

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
                self.assertEqual(actual.reasons, ("unfilled_terminated",))
                self.assertEqual(str(actual.maturity_dates[0]), "2026-01-06")

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
                self.assertTrue(actual.valid_mask[0])
                self.assertEqual(actual.target_raw_r[0], np.float32(expected["r_mult"]))
                self.assertEqual(str(actual.maturity_dates[0]), expected["exit_date"])

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
                self.assertEqual(actual.target_raw_r[0], np.float32(expected["r_mult"]))
                self.assertEqual(str(actual.maturity_dates[0]), expected["exit_date"])

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
                    self.assertEqual(actual.target_raw_r[0], np.float32(baseline.realized_net_r))
                    self.assertEqual(str(actual.maturity_dates[0]), baseline.exit_date)

    def test_no_future_parameter_access_and_midpath_gap_fails_closed(self):
        from filters.breakout_quality import daily_trade40_target as owner
        for prices in ((105, 106, 100, 102, 103, 104), (100, 105, 100, 99, 101, 104)):
            frame, params = self._fixture(prices)
            schedule = self._schedule(frame, params, replace(params, high_len=3),
                                      "rolling_oos_param_schedule")
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


if __name__ == "__main__":
    unittest.main()
