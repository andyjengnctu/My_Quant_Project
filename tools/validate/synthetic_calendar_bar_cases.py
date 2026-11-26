"""Calendar input and masked-fusion invariants in the reusable model suite."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from filters.breakout_quality.calendar_bars import (
    ClosedCalendarBars, calendar_bar_layout, normalize_masked_calendar_windows,
)
from filters.breakout_quality.daily_ranker_data import LazyDailyFeatureBank
from filters.breakout_quality.features import normalize_ohlcv_array_window
from filters.breakout_quality.models.architectures import SUPPORTED_MODEL_ARCHITECTURES
from filters.breakout_quality.models.spec import get_model_spec, model_spec_from_manifest


def check_calendar_bar_inputs(check, check_true):
    calendar = pd.bdate_range("2015-01-01", "2022-02-28")
    # A market-wide closed week must remain a calendar slot, not disappear.
    calendar = calendar[~((calendar >= "2020-12-21") & (calendar <= "2020-12-25"))]
    close = 100.0 + np.arange(len(calendar)) * 0.1
    benchmark = pd.DataFrame({
        "Open": close - 0.5, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 1000.0 + np.arange(len(close)),
    }, index=calendar)
    stock = benchmark.drop(pd.Timestamp("2020-12-30"))
    info = pd.DatetimeIndex(["2021-01-04"])
    week = ClosedCalendarBars(stock, calendar, "W-SUN")
    values, valid = week.take(info, 4)
    check_true("calendar_missing_session_and_empty_market_week_keep_slots", (
        valid.tolist() == [[True, True, False, False]] and not values[0, -2:].any()
    ))
    observed = stock.loc["2020-12-14":"2020-12-20"]
    expected = np.asarray([
        observed["Open"].iloc[0], observed["High"].max(), observed["Low"].min(),
        observed["Close"].iloc[-1], observed["Volume"].sum(),
    ])
    check_true("calendar_week_ohlcv_matches_independent_hand_aggregation", np.array_equal(values[0, 1], expected))
    month = ClosedCalendarBars(stock, calendar, "M")
    month_values, month_valid = month.take(info, 3)
    check_true("calendar_missing_daily_row_masks_whole_month", month_valid.tolist() == [[True, True, False]])
    for freq, dates, expected_ranges in (
        ("W-SUN", ["2020-12-18", "2020-12-28", "2021-01-01", "2021-01-04"],
         [("2020-12-07", "2020-12-13"), None, ("2020-12-21", "2020-12-27"), ("2020-12-28", "2021-01-03")]),
        ("M", ["2020-11-30", "2020-12-01", "2020-12-31", "2021-01-04"],
         [("2020-10-01", "2020-10-31"), ("2020-11-01", "2020-11-30"),
          ("2020-11-01", "2020-11-30"), ("2020-12-01", "2020-12-31")]),
    ):
        bars = ClosedCalendarBars(benchmark, calendar, freq)
        actual, masks = bars.take(pd.DatetimeIndex(dates), 1)
        for index, bounds in enumerate(expected_ranges):
            block = benchmark.iloc[:0] if bounds is None else benchmark.loc[bounds[0]:bounds[1]]
            if block.empty:
                correct = not masks[index, 0] and not actual[index, 0].any()
            else:
                reference = [block["Open"].iloc[0], block["High"].max(), block["Low"].min(), block["Close"].iloc[-1], block["Volume"].sum()]
                correct = bool(masks[index, 0]) and np.array_equal(actual[index, 0], reference)
            check_true(f"calendar_{freq}_{dates[index]}_strict_prior_period", correct)
    early_values, early_valid = month.take(pd.DatetimeIndex(["2015-01-15"]), 60)
    check_true("calendar_short_history_is_all_masked_without_future_backfill", not early_valid.any() and not early_values.any())
    normalized = normalize_masked_calendar_windows(values, valid, np.asarray([stock.loc[info[0], "Close"]]))
    reference = normalize_ohlcv_array_window(values[0, valid[0]], stock.loc[info[0], "Close"])
    check_true("calendar_normalization_uses_only_valid_bars_and_canonical_volume_rule", (
        np.array_equal(normalized[0, valid[0], :5], reference)
        and not normalized[0, ~valid[0]].any()
    ))
    zeros = values.copy()
    zeros[:, :, 4] = 0
    check_true("calendar_valid_zero_volume_remains_observed_and_finite", (
        np.isfinite(normalize_masked_calendar_windows(zeros, valid, np.asarray([100.0]))).all()
        and np.array_equal(normalize_masked_calendar_windows(zeros, valid, np.asarray([100.0]))[:, :, 5], valid)
    ))

    information_dates = pd.DatetimeIndex(["2021-01-04", "2016-08-15", "2020-12-18"])
    def bank(source=stock, market=benchmark, dates=calendar, windows=()):
        return LazyDailyFeatureBank(
            frames=(source,), benchmark=market,
            ticker_ids=np.zeros(len(information_dates), dtype=np.int32),
            source_positions=source.index.get_indexer(information_dates),
            benchmark_positions=market.index.get_indexer(information_dates),
            policy=SimpleNamespace(feature_window_bars=300),
            calendar_bar_windows=windows, exchange_calendar=dates,
        )

    original = bank()[:]
    for architecture in SUPPORTED_MODEL_ARCHITECTURES:
        spec = get_model_spec(architecture)
        if not spec.calendar_bar_windows:
            continue
        layout = calendar_bar_layout(spec.calendar_bar_windows)
        current = bank(windows=spec.calendar_bar_windows)
        features = current[:]
        prefix = architecture + "_"
        check(prefix + "packed_shape", (3, 300, 34), features.shape)
        check_true(prefix + "preserves_original_daily_inputs_and_sample_order", np.array_equal(features[:, :, :10], original))
        for item in layout:
            length, start = item["length"], item["channel_start"]
            check_true(prefix + item["frequency"] + "_leading_transport_is_zero", not features[:, :-length, start:start + 12].any())
        check_true(prefix + "stock_and_benchmark_masks_are_independent", (
            features[0, -1, 15] == 0 and features[0, -1, 21] == 1
            and features[0, -1, 27] == 0 and features[0, -1, 33] == 1
        ))
        check_true(prefix + "short_history_retains_group_and_left_masks", (
            not features[1, :-19, 27].any() and features[1, -1, 27] == 1
        ))
        check_true(prefix + "scalar_and_reordered_batches_agree", np.array_equal(
            current[[2, 0, 2]], np.stack((current[2], current[0], current[2])),
        ))
        prefix_source = stock.loc[:information_dates[0]]
        prefix_market = benchmark.loc[:information_dates[0]]
        prefix_bank = bank(prefix_source, prefix_market, calendar[calendar <= information_dates[0]], spec.calendar_bar_windows)
        check_true(prefix + "future_append_cannot_change_historical_inputs", np.array_equal(current[0], prefix_bank[0]))
        future_source, future_market = stock.copy(), benchmark.copy()
        future_source.loc[future_source.index > information_dates[0], ["Open", "High", "Low", "Close", "Volume"]] *= 10
        future_market.loc[future_market.index > information_dates[0], ["Open", "High", "Low", "Close", "Volume"]] *= 10
        check_true(prefix + "future_value_changes_cannot_change_inputs", np.array_equal(
            current[0], bank(future_source, future_market, windows=spec.calendar_bar_windows)[0],
        ))
        scaled_source, scaled_market = stock.copy(), benchmark.copy()
        scaled_source.loc[:, ["Open", "High", "Low", "Close"]] *= 8
        scaled_market.loc[:, ["Open", "High", "Low", "Close"]] *= 4
        check_true(prefix + "independent_price_scale_invariance", np.array_equal(
            features, bank(scaled_source, scaled_market, windows=spec.calendar_bar_windows)[:],
        ))
        check(prefix + "manifest_roundtrip", spec, model_spec_from_manifest(spec.as_manifest_payload()))
        check_true(prefix + "manifest_matches_actual_channel_order", current.feature_columns == current.manifest_feature_columns)
        check_true(prefix + "window_change_changes_fit_identity", spec.as_manifest_payload() != replace(
            spec, calendar_bar_windows=(("W-SUN", 52), ("M", 36)),
        ).as_manifest_payload())
        from filters.breakout_quality.models.active import build_active_model
        import torch

        torch.manual_seed(42)
        model = build_active_model(features.shape[-1], 0, architecture=architecture).eval()
        x = torch.from_numpy(features)
        context = torch.empty((len(x), 0))
        with torch.no_grad():
            logits = model(x, context)
            contaminated = x.clone()
            for item in layout:
                for offset in (0, 6):
                    start = item["channel_start"] + offset
                    mask = contaminated[:, :, start + 5:start + 6]
                    contaminated[:, :, start:start + 5] += (1 - mask) * 123456.0
            check_true(prefix + "masked_values_cannot_affect_model_output", torch.equal(logits, model(contaminated, context)))
            empty = x.clone()
            empty[:, :, 10:] = 0
            for item, encoder in zip(layout, model.calendar_encoders):
                start = item["channel_start"]
                check_true(prefix + item["frequency"] + "_all_missing_branch_is_zero", not bool(encoder(empty[:, -item["length"]:, start:start + 12]).any()))
        check_true(prefix + "finite_two_logit_ranker_output", tuple(logits.shape) == (3, 2) and bool(torch.isfinite(logits).all()))
        # The same canonical full-list pairwise loss must update both new encoders.
        from services.breakout_quality.train_continuous_ranker import _pairwise_logistic_loss
        from core.breakout_quality_runtime import CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
        before = [encoder.convolutions[0].weight.detach().clone() for encoder in model.calendar_encoders]
        margins = model(x, context).diff(dim=1).squeeze(1)
        loss, pairs = _pairwise_logistic_loss(
            torch, margins, torch.tensor([0.1, 0.5, 0.9]), pd.DatetimeIndex(["2021-01-04"] * 3),
            reduction=CONTINUOUS_RANKER_PAIRWISE_REDUCTION_FULL_LIST_DELTA_NDCG,
        )
        loss.backward()
        optimizer.step()
        check_true(prefix + "canonical_pairwise_loss_updates_all_calendar_branches", (
            pairs > 0 and bool(torch.isfinite(loss)) and all(
                not torch.equal(old, encoder.convolutions[0].weight.detach())
                for old, encoder in zip(before, model.calendar_encoders)
            )
        ))
        restored = build_active_model(features.shape[-1], 0, architecture=architecture).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            check_true(prefix + "checkpoint_state_roundtrip_preserves_predictions", torch.equal(
                model(x, context), restored(x, context),
            ))
