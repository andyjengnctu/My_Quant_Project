"""Ordered Daily sample collection; target computation stays with its owner.

Target-valid rows retain their original ticker/date order. Inference-only
rows, including tickers without a valid Target, are appended afterward.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DailyTickerSamples:
    """One already evaluated ticker; collection cannot change eligibility."""

    ticker: str
    frame: pd.DataFrame
    local_positions: np.ndarray
    local_benchmark_positions: np.ndarray
    frame_dates: pd.DatetimeIndex
    local_target_valid: np.ndarray
    local_targets: np.ndarray
    local_context: np.ndarray
    local_favorable: np.ndarray
    local_adverse: np.ndarray
    local_opportunity_bar: np.ndarray
    local_first_breach_bar: np.ndarray
    local_minimum_low_return: np.ndarray
    local_label_end_dates: np.ndarray | None
    local_standard_price_mature: np.ndarray
    local_trade_r_diagnostic: np.ndarray | None


class DailySampleAccumulator:
    """Own valid/inference partitions and the stable ticker identity mapping."""

    def __init__(self, *, sample_universe_horizon: int, trade_path_mode: bool,
                 trade_diagnostic_r_required: bool):
        self.sample_universe_horizon = sample_universe_horizon
        self.trade_path_mode = trade_path_mode
        self.trade_diagnostic_r_required = trade_diagnostic_r_required
        self.tickers: list[str] = []
        self.frames: list[pd.DataFrame] = []

        # Keep target-valid rows first in the exact historical ticker/date order.  This preserves
        # the scientific training universe while allowing additional inference-only rows to be
        # appended without making score availability depend on future target completion.
        self.valid_ticker_id_chunks: list[np.ndarray] = []
        self.valid_source_pos_chunks: list[np.ndarray] = []
        self.valid_benchmark_pos_chunks: list[np.ndarray] = []
        self.valid_date_chunks: list[np.ndarray] = []
        self.valid_label_end_chunks: list[np.ndarray] = []
        self.valid_target_chunks: list[np.ndarray] = []
        # Only money-denominated targets need a separate, canonical R diagnostic.
        self.valid_trade_r_diagnostic_chunks: list[np.ndarray] = []
        self.valid_context_chunks: list[np.ndarray] = []
        self.valid_favorable_chunks: list[np.ndarray] = []
        self.valid_adverse_chunks: list[np.ndarray] = []
        self.valid_opportunity_bar_chunks: list[np.ndarray] = []
        self.valid_first_breach_bar_chunks: list[np.ndarray] = []
        self.valid_opportunity_date_chunks: list[np.ndarray] = []
        self.valid_first_breach_date_chunks: list[np.ndarray] = []
        self.valid_minimum_low_return_chunks: list[np.ndarray] = []
        self.valid_standard_price_mature_chunks: list[np.ndarray] = []

        self.inference_ticker_id_chunks: list[np.ndarray] = []
        self.inference_source_pos_chunks: list[np.ndarray] = []
        self.inference_benchmark_pos_chunks: list[np.ndarray] = []
        self.inference_date_chunks: list[np.ndarray] = []
        self.inference_context_chunks: list[np.ndarray] = []
        # Common 40D prices have already been computed by the canonical target
        # batch before valid/NULL rows are separated. Carry the inference-only
        # values forward instead of recomputing the same 40-session windows after
        # constructing the group table (potentially hundreds of thousands of rows).
        self.inference_price_favorable_chunks: list[np.ndarray] = []
        self.inference_price_adverse_chunks: list[np.ndarray] = []
        self.pending_inference_only_tickers: list[tuple] = []

    def append(self, sample: DailyTickerSamples) -> None:
        ticker = sample.ticker
        frame = sample.frame
        local_positions = sample.local_positions
        local_benchmark_positions = sample.local_benchmark_positions
        frame_dates = sample.frame_dates
        local_target_valid = sample.local_target_valid
        local_targets = sample.local_targets
        local_context = sample.local_context
        local_favorable = sample.local_favorable
        local_adverse = sample.local_adverse
        local_opportunity_bar = sample.local_opportunity_bar
        local_first_breach_bar = sample.local_first_breach_bar
        local_minimum_low_return = sample.local_minimum_low_return
        local_label_end_dates = sample.local_label_end_dates
        local_standard_price_mature = sample.local_standard_price_mature
        local_trade_r_diagnostic = sample.local_trade_r_diagnostic
        sample_universe_horizon = self.sample_universe_horizon
        trade_path_mode = self.trade_path_mode
        trade_diagnostic_r_required = self.trade_diagnostic_r_required
        valid_positions = local_positions[local_target_valid]
        valid_benchmark_positions = local_benchmark_positions[local_target_valid]
        valid_targets = local_targets[local_target_valid]
        valid_context = local_context[local_target_valid]
        valid_favorable = local_favorable[local_target_valid]
        valid_adverse = local_adverse[local_target_valid]
        valid_opportunity_bar = local_opportunity_bar[local_target_valid]
        valid_first_breach_bar = local_first_breach_bar[local_target_valid]
        valid_minimum_low_return = local_minimum_low_return[local_target_valid]
        invalid_positions = local_positions[~local_target_valid]
        invalid_benchmark_positions = local_benchmark_positions[~local_target_valid]
        invalid_context = local_context[~local_target_valid]

        if len(valid_positions):
            ticker_id = len(self.tickers)
            self.tickers.append(ticker)
            self.frames.append(frame)
            count = len(valid_positions)
            self.valid_ticker_id_chunks.append(np.full(count, ticker_id, dtype=np.int32))
            self.valid_source_pos_chunks.append(np.asarray(valid_positions, dtype=np.int32))
            self.valid_benchmark_pos_chunks.append(
                np.asarray(valid_benchmark_positions, dtype=np.int32)
            )
            self.valid_date_chunks.append(
                frame_dates.take(valid_positions).to_numpy(dtype="datetime64[D]")
            )
            self.valid_label_end_chunks.append(
                np.asarray(local_label_end_dates[local_target_valid], dtype="datetime64[D]")
                if local_label_end_dates is not None else
                frame_dates.take(valid_positions + sample_universe_horizon).to_numpy(
                    dtype="datetime64[D]"
                )
            )
            self.valid_target_chunks.append(np.asarray(valid_targets, dtype=np.float32))
            if trade_diagnostic_r_required:
                self.valid_trade_r_diagnostic_chunks.append(np.asarray(
                    local_trade_r_diagnostic[local_target_valid], dtype=np.float32,
                ))
            self.valid_context_chunks.append(np.asarray(valid_context, dtype=np.float32))
            self.valid_favorable_chunks.append(np.asarray(valid_favorable, dtype=np.float32))
            self.valid_adverse_chunks.append(np.asarray(valid_adverse, dtype=np.float32))
            self.valid_opportunity_bar_chunks.append(np.asarray(valid_opportunity_bar, dtype=np.int16))
            self.valid_first_breach_bar_chunks.append(np.asarray(valid_first_breach_bar, dtype=np.int16))
            opportunity_dates = np.full(
                len(valid_positions), np.datetime64("NaT"), dtype="datetime64[D]"
            )
            opportunity_mask = np.asarray(valid_opportunity_bar, dtype=np.int64) >= 1
            if bool(np.any(opportunity_mask)):
                opportunity_positions = (
                    valid_positions[opportunity_mask]
                    + np.asarray(valid_opportunity_bar, dtype=np.int64)[opportunity_mask]
                )
                opportunity_dates[opportunity_mask] = frame_dates.take(
                    opportunity_positions
                ).to_numpy(dtype="datetime64[D]")
            self.valid_opportunity_date_chunks.append(opportunity_dates)
            breach_dates = np.full(len(valid_positions), np.datetime64("NaT"), dtype="datetime64[D]")
            breach_mask = np.asarray(valid_first_breach_bar, dtype=np.int64) >= 1
            if bool(np.any(breach_mask)):
                breach_positions = (
                    valid_positions[breach_mask]
                    + np.asarray(valid_first_breach_bar, dtype=np.int64)[breach_mask]
                )
                breach_dates[breach_mask] = frame_dates.take(breach_positions).to_numpy(
                    dtype="datetime64[D]"
                )
            self.valid_first_breach_date_chunks.append(breach_dates)
            self.valid_minimum_low_return_chunks.append(np.asarray(valid_minimum_low_return, dtype=np.float32))
            if trade_path_mode:
                self.valid_standard_price_mature_chunks.append(
                    np.asarray(local_standard_price_mature[local_target_valid], dtype=bool)
                )
            if len(invalid_positions):
                self.inference_ticker_id_chunks.append(
                    np.full(len(invalid_positions), ticker_id, dtype=np.int32)
                )
                self.inference_source_pos_chunks.append(
                    np.asarray(invalid_positions, dtype=np.int32)
                )
                self.inference_benchmark_pos_chunks.append(
                    np.asarray(invalid_benchmark_positions, dtype=np.int32)
                )
                self.inference_date_chunks.append(
                    frame_dates.take(invalid_positions).to_numpy(dtype="datetime64[D]")
                )
                self.inference_context_chunks.append(np.asarray(invalid_context, dtype=np.float32))
                if trade_path_mode:
                    self.inference_price_favorable_chunks.append(np.asarray(
                        local_favorable[~local_target_valid], dtype=np.float32,
                    ))
                    self.inference_price_adverse_chunks.append(np.asarray(
                        local_adverse[~local_target_valid], dtype=np.float32,
                    ))
        else:
            # A ticker with no target-valid row can still be score-eligible.  Append these
            # tickers after the historical training tickers so existing target-valid ticker
            # identities and group ordering remain stable.
            self.pending_inference_only_tickers.append(
                (
                    ticker,
                    frame,
                    np.asarray(invalid_positions, dtype=np.int32),
                    np.asarray(invalid_benchmark_positions, dtype=np.int32),
                    frame_dates,
                    np.asarray(invalid_context, dtype=np.float32),
                    np.asarray(local_favorable[~local_target_valid], dtype=np.float32) if trade_path_mode else None,
                    np.asarray(local_adverse[~local_target_valid], dtype=np.float32) if trade_path_mode else None,
                )
            )

    def finish_inference_only_tickers(self) -> None:
        trade_path_mode = self.trade_path_mode
        for (ticker, frame, positions, benchmark_positions_local, frame_dates,
             pending_context, pending_favorable, pending_adverse) in self.pending_inference_only_tickers:
            if len(positions) == 0:
                continue
            ticker_id = len(self.tickers)
            self.tickers.append(ticker)
            self.frames.append(frame)
            self.inference_ticker_id_chunks.append(
                np.full(len(positions), ticker_id, dtype=np.int32)
            )
            self.inference_source_pos_chunks.append(np.asarray(positions, dtype=np.int32))
            self.inference_benchmark_pos_chunks.append(
                np.asarray(benchmark_positions_local, dtype=np.int32)
            )
            self.inference_date_chunks.append(
                frame_dates.take(positions).to_numpy(dtype="datetime64[D]")
            )
            self.inference_context_chunks.append(np.asarray(pending_context, dtype=np.float32))
            if trade_path_mode:
                self.inference_price_favorable_chunks.append(np.asarray(pending_favorable, dtype=np.float32))
                self.inference_price_adverse_chunks.append(np.asarray(pending_adverse, dtype=np.float32))

        self.pending_inference_only_tickers.clear()
