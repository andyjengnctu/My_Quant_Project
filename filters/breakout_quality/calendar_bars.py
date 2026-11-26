"""Closed-calendar OHLCV inputs derived from the canonical daily source.

The transport is one ordinary feature tensor. Calendar sequences occupy their
own channels, right aligned within that tensor; model branches decode only
their declared lengths. This keeps batching, fitting and inference generic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from filters.breakout_quality.features import normalize_ohlcv_array_windows


_PERIOD_NAMES = {"W-SUN": "Week", "M": "Month"}
_FIELDS = ("Open", "High", "Low", "Close", "Volume", "Valid")
DAILY_CHANNELS = 2 * (len(_FIELDS) - 1)
CALENDAR_CHANNELS = 2 * len(_FIELDS)


def calendar_bar_layout(windows: tuple[tuple[str, int], ...]) -> tuple[dict, ...]:
    if not windows or len({freq for freq, _ in windows}) != len(windows):
        raise ValueError("Calendar windows must be nonempty and unique")
    layout = []
    for index, (freq, length) in enumerate(windows):
        if freq not in _PERIOD_NAMES or type(length) is not int or length < 1:
            raise ValueError("Unsupported calendar frequency or window length")
        layout.append({
            "frequency": freq, "length": length,
            "channel_start": DAILY_CHANNELS + index * CALENDAR_CHANNELS,
            "feature_columns": [
                f"{_PERIOD_NAMES[freq]}_{instrument}_{field}"
                for instrument in ("Stock", "Benchmark") for field in _FIELDS
            ],
        })
    return tuple(layout)


def calendar_bar_contract(windows: tuple[tuple[str, int], ...]) -> dict:
    return {
        "contract_id": "closed_calendar_ohlcv_masked_v1",
        "windows": list(calendar_bar_layout(windows)),
        "source": "complete_canonical_sanitized_adjusted_daily_history",
        "calendar": "same_canonical_exchange_sessions_as_daily_target",
        "availability": "strictly_before_information_date_calendar_period",
        "aggregation": "first_open_max_high_min_low_last_close_sum_volume",
        "validity": "all_expected_exchange_sessions_present_in_instrument",
        "calendar_history_boundary": "first_period_masked_if_calendar_starts_inside_period",
        "empty_or_incomplete_period": "preserve_calendar_slot_zero_values_valid_mask_zero",
        "history_shortfall": "left_pad_missing_calendar_slots_without_excluding_sample",
        "prices": "divide_by_instrument_information_date_close_minus_one",
        "volume": "canonical_log1p_median_iqr_on_valid_bars_in_each_window",
        "packing": "daily_channels_first_calendar_channels_right_aligned_zero_leading_transport",
        "dtype": "float64_aggregation_float32_normalized_input",
    }


class ClosedCalendarBars:
    """Immutable aggregates; lookup never compresses a missing calendar period."""

    def __init__(self, frame: pd.DataFrame, calendar: pd.DatetimeIndex, frequency: str):
        dates = pd.DatetimeIndex(frame.index).normalize()
        if (dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing
                or not dates.difference(calendar).empty):
            raise ValueError("Calendar OHLCV requires sorted canonical dates in the exchange calendar")
        values = frame[list(_FIELDS[:-1])].to_numpy(dtype=np.float64)
        if (not np.isfinite(values).all() or np.any(values[:, :4] <= 0)
                or np.any(values[:, 4] < 0)
                or np.any(values[:, 1] < values[:, :4].max(axis=1))
                or np.any(values[:, 2] > values[:, :4].min(axis=1))):
            raise ValueError("Calendar OHLCV requires sanitized daily values")
        self.frequency = frequency
        periods = pd.period_range(calendar[0], calendar[-1], freq=frequency)
        self._first_ordinal = int(periods.asi8[0])
        groups = frame.groupby(dates.to_period(frequency), sort=True)
        aggregated = groups.agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).reindex(periods)
        counts = groups.size().reindex(periods, fill_value=0).to_numpy()
        expected = pd.Series(1, index=calendar.to_period(frequency)).groupby(level=0).sum()
        expected_counts = expected.reindex(periods, fill_value=0).to_numpy()
        self._valid = ((expected_counts > 0) & (counts == expected_counts)
                       & (periods.start_time >= calendar[0]))
        self._values = aggregated.to_numpy(dtype=np.float64)
        self._values[~self._valid] = 0.0

    def take(self, information_dates: pd.DatetimeIndex, length: int):
        # Even Friday/month-end close waits until the following calendar period.
        current = information_dates.to_period(self.frequency).asi8
        positions = current[:, None] - np.arange(length, 0, -1) - self._first_ordinal
        in_range = (positions >= 0) & (positions < len(self._values))
        clipped = np.clip(positions, 0, len(self._values) - 1)
        valid = in_range & self._valid[clipped]
        values = self._values[clipped].copy()
        values[~valid] = 0.0
        return values, valid


def normalize_masked_calendar_windows(values, valid, anchors) -> np.ndarray:
    """Apply the existing numerical owner to valid bars, then restore slots."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    anchors = np.asarray(anchors, dtype=np.float64)
    if values.shape != (*valid.shape, 5) or anchors.shape != (len(values),):
        raise ValueError("Calendar normalization shape mismatch")
    if not np.isfinite(anchors).all() or np.any(anchors <= 0):
        raise ValueError("Calendar normalization requires finite positive information-date closes")
    result = np.zeros((*valid.shape, len(_FIELDS)), dtype=np.float32)
    result[:, :, -1] = valid
    counts = valid.sum(axis=1)
    # Group by count to batch the canonical normalizer without fabricated bars
    # entering volume quantiles. Missing positions remain in the output grid.
    for count in np.unique(counts[counts > 0]):
        rows = np.flatnonzero(counts == count)
        masks = valid[rows]
        observed = values[rows][masks].reshape(len(rows), int(count), 5)
        normalized = normalize_ohlcv_array_windows(observed, anchors[rows])
        selected = result[rows]
        selected[masks, :5] = normalized.reshape(-1, 5)
        result[rows] = selected
    return result


class CalendarBarFeatureBank:
    """Preaggregate per instrument once; materialize only requested batch windows."""

    def __init__(self, *, frames, benchmark, exchange_calendar, windows, sequence_length):
        self.layout = calendar_bar_layout(windows)
        if any(item["length"] > sequence_length for item in self.layout):
            raise ValueError("Calendar branch exceeds the packed sequence length")
        calendar = pd.DatetimeIndex(exchange_calendar).normalize()
        if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("Calendar inputs require the canonical sorted exchange calendar")
        self._frames = frames
        self._benchmark = benchmark
        self._dates = tuple(pd.DatetimeIndex(frame.index).normalize() for frame in frames)
        self._benchmark_dates = pd.DatetimeIndex(benchmark.index).normalize()
        self._stock_bars = tuple(
            tuple(ClosedCalendarBars(frame, calendar, item["frequency"]) for frame in frames)
            for item in self.layout
        )
        self._benchmark_bars = tuple(
            ClosedCalendarBars(benchmark, calendar, item["frequency"]) for item in self.layout
        )
        self.feature_columns = tuple(name for item in self.layout for name in item["feature_columns"])

    def pack(self, output, ticker_ids, source_positions, benchmark_positions):
        output[:, :, DAILY_CHANNELS:] = 0.0
        for item, stock_bars, benchmark_bars in zip(
            self.layout, self._stock_bars, self._benchmark_bars, strict=True,
        ):
            length, start = item["length"], item["channel_start"]
            values = np.zeros((len(output), length, 5), dtype=np.float64)
            valid = np.zeros((len(output), length), dtype=bool)
            anchors = np.empty(len(output), dtype=np.float64)
            for ticker_id in np.unique(ticker_ids):
                rows = np.flatnonzero(ticker_ids == ticker_id)
                positions = source_positions[rows]
                values[rows], valid[rows] = stock_bars[int(ticker_id)].take(
                    self._dates[int(ticker_id)].take(positions), length,
                )
                anchors[rows] = self._frames[int(ticker_id)]["Close"].to_numpy()[positions]
            output[:, -length:, start:start + len(_FIELDS)] = normalize_masked_calendar_windows(values, valid, anchors)
            unique, inverse = np.unique(benchmark_positions, return_inverse=True)
            values, valid = benchmark_bars.take(self._benchmark_dates.take(unique), length)
            normalized = normalize_masked_calendar_windows(
                values, valid, self._benchmark["Close"].to_numpy()[unique],
            )
            output[:, -length:, start + len(_FIELDS):start + CALENDAR_CHANNELS] = normalized[inverse]
