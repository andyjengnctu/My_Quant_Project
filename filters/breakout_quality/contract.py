"""Breakout quality filter contract and immutable label policy."""

from __future__ import annotations

from dataclasses import dataclass, asdict

FILTER_FAMILY = "breakout_quality"
DEFAULT_FILTER_ID = "breakout_quality_v1"
DEFAULT_SCORE_FILENAME = "scores.csv"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MODEL_FILENAME = "model.pt"
DEFAULT_FEATURE_WINDOW_BARS = 60
DEFAULT_LABEL_HORIZON_BARS = 40
DEFAULT_HIGH_LEN_MIN = 100
DEFAULT_HIGH_LEN_MAX = 300
DEFAULT_HIGH_LEN_STEP = 5
DEFAULT_BENCHMARK_TICKER = "0050"


@dataclass(frozen=True)
class BreakoutQualityLabelPolicy:
    feature_window_bars: int = DEFAULT_FEATURE_WINDOW_BARS
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS
    high_len_min: int = DEFAULT_HIGH_LEN_MIN
    high_len_max: int = DEFAULT_HIGH_LEN_MAX
    high_len_step: int = DEFAULT_HIGH_LEN_STEP
    label_atr_len: int = 14
    label_atr_buy_tol: float = 1.5
    label_atr_times_init: float = 2.0
    positive_mfe_r: float = 1.5
    negative_mae_r: float = -1.0
    reject_confirm_mfe_r: float = 1.0
    dead_mfe_r: float = 0.5
    evaluate_from_bars_after_entry: int = 1
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER

    def high_lens(self) -> tuple[int, ...]:
        return tuple(range(int(self.high_len_min), int(self.high_len_max) + 1, int(self.high_len_step)))

    def as_manifest_payload(self) -> dict:
        return asdict(self)


DEFAULT_LABEL_POLICY = BreakoutQualityLabelPolicy()

LABEL_REJECT = 0
LABEL_PASS = 1
LABEL_IGNORE = -1
LABEL_NAME_MAP = {
    LABEL_IGNORE: "IGNORE",
    LABEL_REJECT: "REJECT",
    LABEL_PASS: "PASS",
}

FEATURE_COLUMNS = (
    "open_norm",
    "high_norm",
    "low_norm",
    "close_norm",
    "volume_norm",
    "benchmark_open_norm",
    "benchmark_high_norm",
    "benchmark_low_norm",
    "benchmark_close_norm",
    "benchmark_volume_norm",
)

CONTEXT_COLUMNS = (
    "high_len_norm",
    "breakout_level_to_close",
    "close_to_breakout_level",
    "high_to_breakout_level",
)


__all__ = [
    "CONTEXT_COLUMNS",
    "DEFAULT_FILTER_ID",
    "DEFAULT_LABEL_POLICY",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_SCORE_FILENAME",
    "FEATURE_COLUMNS",
    "FILTER_FAMILY",
    "LABEL_IGNORE",
    "LABEL_NAME_MAP",
    "LABEL_PASS",
    "LABEL_REJECT",
    "BreakoutQualityLabelPolicy",
]
