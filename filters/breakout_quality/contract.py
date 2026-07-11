"""Breakout quality filter artifact, feature, and label contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from config.breakout_quality_policy import (
    BREAKOUT_QUALITY_BENCHMARK_TICKER,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
    BREAKOUT_QUALITY_LABEL_HORIZON_BARS,
    BREAKOUT_QUALITY_LABEL_PASS_RETURN,
    BREAKOUT_QUALITY_LABEL_REJECT_RETURN,
    build_breakout_quality_default_high_len_values,
)

FILTER_FAMILY = "breakout_quality"
DEFAULT_FILTER_ID = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
DEFAULT_SCORE_FILENAME = "scores.csv"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MODEL_FILENAME = "model.pt"
DEFAULT_SPLIT_FILENAME = "split_assignments.csv"
DEFAULT_FEATURE_WINDOW_BARS = BREAKOUT_QUALITY_FEATURE_WINDOW_BARS
DEFAULT_LABEL_HORIZON_BARS = BREAKOUT_QUALITY_LABEL_HORIZON_BARS
DEFAULT_BENCHMARK_TICKER = BREAKOUT_QUALITY_BENCHMARK_TICKER
ARTIFACT_CONTRACT_VERSION = 5
SCORE_TABLE_SCHEMA_VERSION = 2
SPLIT_ASSIGNMENT_SCHEMA_VERSION = 2
SCORE_COLUMN = "dl_quality_score"
SCORE_COMPARISON = ">="
SCORE_THRESHOLD_SOURCE = "strategy_param.breakout_quality_score_threshold"
RUNTIME_SCOPE_RESEARCH = "research"
RUNTIME_SCOPE_NOT_EXPORTED = "not_exported"
RUNTIME_SCOPE_FORWARD_OOS = "forward_oos"
RUNTIME_SCOPE_ROLLING_OOS = "rolling_oos"
RUNTIME_ELIGIBLE_SCOPES = frozenset({RUNTIME_SCOPE_FORWARD_OOS})
TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION = "fixed_epoch_full_selection"
TRAINING_MODE_INNER_VALIDATION_FULL_REFIT = "inner_validation_epoch_selection_full_refit"
OUTER_SPLIT_SELECTION = "selection"
OUTER_SPLIT_OOS = "oos"
OUTER_SPLIT_OUT_OF_SCOPE = "out_of_scope"
OUTER_SPLIT_VALUES = (
    OUTER_SPLIT_SELECTION,
    OUTER_SPLIT_OOS,
    OUTER_SPLIT_OUT_OF_SCOPE,
)
SELECTION_ROLE_TRAIN = "train"
SELECTION_ROLE_VALIDATION = "validation"
SELECTION_ROLE_INNER_EMBARGO = "inner_embargo"
SELECTION_ROLE_EMBARGO = "embargo"
SELECTION_ROLE_IGNORE = "ignore"
SELECTION_ROLE_NOT_APPLICABLE = "not_applicable"
SELECTION_ROLE_VALUES = (
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_IGNORE,
    SELECTION_ROLE_NOT_APPLICABLE,
)



@dataclass(frozen=True)
class BreakoutQualityLabelPolicy:
    feature_window_bars: int = DEFAULT_FEATURE_WINDOW_BARS
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS
    high_len_values: tuple[int, ...] = field(default_factory=build_breakout_quality_default_high_len_values)
    pass_return_threshold: float = BREAKOUT_QUALITY_LABEL_PASS_RETURN
    reject_return_threshold: float = BREAKOUT_QUALITY_LABEL_REJECT_RETURN
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER

    def __post_init__(self) -> None:
        if int(self.feature_window_bars) < 1 or int(self.label_horizon_bars) < 1:
            raise ValueError("feature window 與 label horizon 必須 >= 1")
        if float(self.pass_return_threshold) <= 0.0:
            raise ValueError("pass_return_threshold 必須 > 0")
        if not -1.0 < float(self.reject_return_threshold) < 0.0:
            raise ValueError("reject_return_threshold 必須介於 -1 與 0 之間")
        if not str(self.benchmark_ticker).strip():
            raise ValueError("benchmark_ticker 不可空白")
        self.high_lens()

    def high_lens(self) -> tuple[int, ...]:
        normalized = tuple(sorted({int(value) for value in self.high_len_values}))
        if not normalized or normalized[0] < 1:
            raise ValueError("breakout quality high_len_values 必須包含至少一個正整數")
        return normalized

    @property
    def high_len_min(self) -> int:
        return int(self.high_lens()[0])

    @property
    def high_len_max(self) -> int:
        return int(self.high_lens()[-1])

    def as_manifest_payload(self) -> dict:
        payload = asdict(self)
        payload["high_len_values"] = list(self.high_lens())
        payload["high_len_min"] = self.high_len_min
        payload["high_len_max"] = self.high_len_max
        return payload


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

SPLIT_ASSIGNMENT_REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "high_len",
    "outer_split",
    "selection_role",
)

SCORE_TABLE_REQUIRED_COLUMNS = (
    "ticker",
    "date",
    "high_len",
    SCORE_COLUMN,
)


__all__ = [
    "ARTIFACT_CONTRACT_VERSION",
    "CONTEXT_COLUMNS",
    "DEFAULT_FILTER_ID",
    "DEFAULT_LABEL_POLICY",
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_MODEL_FILENAME",
    "DEFAULT_SCORE_FILENAME",
    "DEFAULT_SPLIT_FILENAME",
    "FEATURE_COLUMNS",
    "FILTER_FAMILY",
    "LABEL_IGNORE",
    "LABEL_NAME_MAP",
    "LABEL_PASS",
    "LABEL_REJECT",
    "OUTER_SPLIT_OOS",
    "OUTER_SPLIT_OUT_OF_SCOPE",
    "OUTER_SPLIT_SELECTION",
    "OUTER_SPLIT_VALUES",
    "SCORE_COLUMN",
    "SCORE_COMPARISON",
    "SCORE_THRESHOLD_SOURCE",
    "RUNTIME_ELIGIBLE_SCOPES",
    "RUNTIME_SCOPE_FORWARD_OOS",
    "RUNTIME_SCOPE_NOT_EXPORTED",
    "RUNTIME_SCOPE_RESEARCH",
    "RUNTIME_SCOPE_ROLLING_OOS",
    "SCORE_TABLE_REQUIRED_COLUMNS",
    "SELECTION_ROLE_EMBARGO",
    "SELECTION_ROLE_IGNORE",
    "SELECTION_ROLE_INNER_EMBARGO",
    "SELECTION_ROLE_NOT_APPLICABLE",
    "SELECTION_ROLE_TRAIN",
    "SELECTION_ROLE_VALIDATION",
    "SELECTION_ROLE_VALUES",
    "SPLIT_ASSIGNMENT_REQUIRED_COLUMNS",
    "SPLIT_ASSIGNMENT_SCHEMA_VERSION",
    "TRAINING_MODE_FIXED_EPOCH_FULL_SELECTION",
    "TRAINING_MODE_INNER_VALIDATION_FULL_REFIT",
    "SCORE_TABLE_SCHEMA_VERSION",
    "BreakoutQualityLabelPolicy",
]
