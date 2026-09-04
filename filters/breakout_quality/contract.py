"""Breakout quality filter artifact, feature, and label contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from core.breakout_quality_registry import (
    get_breakout_quality_experiment_profile,
    normalize_breakout_quality_experiment_profile,
)
from filters.breakout_quality.models.spec import normalize_active_model_architecture

from config.breakout_quality import (
    BREAKOUT_QUALITY_BENCHMARK_TICKER,
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE,
    BREAKOUT_QUALITY_DEFAULT_FILTER_ID,
    BREAKOUT_QUALITY_FEATURE_WINDOW_BARS,
    BREAKOUT_QUALITY_LABEL_HORIZON_BARS,
    BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN,
    BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO,
    BREAKOUT_QUALITY_MODEL_ARCHITECTURE,
    BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS,
)
from core.breakout_quality_policy import (
    build_breakout_quality_default_high_len_values,
)

FILTER_FAMILY = "breakout_quality"
DEFAULT_FILTER_ID = BREAKOUT_QUALITY_DEFAULT_FILTER_ID
DEFAULT_SCORE_FILENAME = "scores.csv"
DEFAULT_UNAVAILABLE_SCORE_FILENAME = "unavailable_scores.csv"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_MODEL_FILENAME = "model.pt"
DEFAULT_SPLIT_FILENAME = "split_assignments.csv"
DEFAULT_FEATURE_WINDOW_BARS = BREAKOUT_QUALITY_FEATURE_WINDOW_BARS
DEFAULT_LABEL_HORIZON_BARS = BREAKOUT_QUALITY_LABEL_HORIZON_BARS
DEFAULT_BENCHMARK_TICKER = BREAKOUT_QUALITY_BENCHMARK_TICKER
DEFAULT_MODEL_ARCHITECTURE = normalize_active_model_architecture(BREAKOUT_QUALITY_MODEL_ARCHITECTURE)
DEFAULT_EXPERIMENT_PROFILE = normalize_breakout_quality_experiment_profile(
    BREAKOUT_QUALITY_EXPERIMENT_PROFILE
)
DEFAULT_EXPERIMENT_SETTINGS = get_breakout_quality_experiment_profile(
    DEFAULT_EXPERIMENT_PROFILE
)
ARTIFACT_CONTRACT_VERSION = 10
SCORE_TABLE_SCHEMA_VERSION = 2
SPLIT_ASSIGNMENT_SCHEMA_VERSION = 3
LABEL_OBJECTIVE = "binary_risk_adjusted_opportunity_v2"
TRADE_PATH_LABEL_OBJECTIVE = "binary_a2_realized_trade_path_v1"
TRADE_PATH_FILTER_ID = "breakout_quality_a2_trade_path_v1"
TRADE_PATH_LABEL_CONTRACT_VERSION = 2
TRADE_PATH_LABEL_STATUS_PASS = "PASS"
TRADE_PATH_LABEL_STATUS_REJECT = "REJECT"
TRADE_PATH_LABEL_STATUS_EXCLUDED = "EXCLUDED"
LEGACY_LABEL_OBJECTIVE = "binary_pass_vs_not_pass_v1"
SCORE_COLUMN = "dl_quality_score"
SCORE_COMPARISON = ">="
SCORE_THRESHOLD_SOURCE = "strategy_param.breakout_quality_score_threshold"
RUNTIME_SCOPE_RESEARCH = "research"
RUNTIME_SCOPE_NOT_EXPORTED = "not_exported"
RUNTIME_SCOPE_FORWARD_OOS = "forward_oos"
RUNTIME_SCOPE_WORKFLOW = "workflow_runtime"
RUNTIME_SCOPE_ROLLING_OOS = "rolling_oos"
RUNTIME_ELIGIBLE_SCOPES = frozenset({RUNTIME_SCOPE_FORWARD_OOS, RUNTIME_SCOPE_WORKFLOW})
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
SELECTION_ROLE_INVALID = "invalid"
SELECTION_ROLE_NOT_APPLICABLE = "not_applicable"
SELECTION_ROLE_VALUES = (
    SELECTION_ROLE_TRAIN,
    SELECTION_ROLE_VALIDATION,
    SELECTION_ROLE_INNER_EMBARGO,
    SELECTION_ROLE_EMBARGO,
    SELECTION_ROLE_INVALID,
    SELECTION_ROLE_NOT_APPLICABLE,
)



@dataclass(frozen=True)
class BreakoutQualityLabelPolicy:
    feature_window_bars: int = DEFAULT_FEATURE_WINDOW_BARS
    label_horizon_bars: int = DEFAULT_LABEL_HORIZON_BARS
    label_path_cache_bars: int = BREAKOUT_QUALITY_LABEL_PATH_CACHE_BARS
    high_len_values: tuple[int, ...] = field(default_factory=build_breakout_quality_default_high_len_values)
    min_mfe_return: float = BREAKOUT_QUALITY_LABEL_MIN_MFE_RETURN
    min_reward_risk_ratio: float = BREAKOUT_QUALITY_LABEL_MIN_REWARD_RISK_RATIO
    max_adverse_return: float = BREAKOUT_QUALITY_LABEL_MAX_ADVERSE_RETURN
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER

    def __post_init__(self) -> None:
        if int(self.feature_window_bars) < 1 or int(self.label_horizon_bars) < 1:
            raise ValueError("feature window 與 label horizon 必須 >= 1")
        if int(self.label_path_cache_bars) < int(self.label_horizon_bars):
            raise ValueError("label_path_cache_bars 必須 >= label_horizon_bars")
        if float(self.min_mfe_return) <= 0.0:
            raise ValueError("min_mfe_return 必須 > 0")
        if float(self.min_reward_risk_ratio) <= 1.0:
            raise ValueError("min_reward_risk_ratio 必須 > 1")
        if not -1.0 < float(self.max_adverse_return) < 0.0:
            raise ValueError("max_adverse_return 必須介於 -1 與 0 之間")
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


    def feature_cache_manifest_payload(self) -> dict:
        return {
            "feature_window_bars": int(self.feature_window_bars),
            "label_path_cache_bars": int(self.label_path_cache_bars),
            "high_len_values": list(self.high_lens()),
            "high_len_min": self.high_len_min,
            "high_len_max": self.high_len_max,
            "benchmark_ticker": str(self.benchmark_ticker),
        }

    def label_manifest_payload(self) -> dict:
        return {
            "label_objective": LABEL_OBJECTIVE,
            "label_horizon_bars": int(self.label_horizon_bars),
            "min_mfe_return": float(self.min_mfe_return),
            "min_reward_risk_ratio": float(self.min_reward_risk_ratio),
            "max_adverse_return": float(self.max_adverse_return),
        }

    def as_manifest_payload(self) -> dict:
        payload = asdict(self)
        payload["label_objective"] = LABEL_OBJECTIVE
        payload["high_len_values"] = list(self.high_lens())
        payload["high_len_min"] = self.high_len_min
        payload["high_len_max"] = self.high_len_max
        return payload


def label_manifest_payload_from_policy_manifest(policy: Mapping[str, object]) -> dict:
    """Return the label-only payload for current or legacy dataset metadata."""

    objective = str(policy.get("label_objective") or "").strip()
    if not objective:
        if "pass_return_threshold" in policy and "reject_return_threshold" in policy:
            objective = LEGACY_LABEL_OBJECTIVE
        elif {"min_mfe_return", "min_reward_risk_ratio", "max_adverse_return"}.issubset(policy):
            objective = LABEL_OBJECTIVE

    if objective == LEGACY_LABEL_OBJECTIVE:
        return {
            "label_objective": LEGACY_LABEL_OBJECTIVE,
            "label_horizon_bars": policy.get("label_horizon_bars"),
            "pass_return_threshold": policy.get("pass_return_threshold"),
            "reject_return_threshold": policy.get("reject_return_threshold"),
        }
    if objective == LABEL_OBJECTIVE:
        return {
            "label_objective": LABEL_OBJECTIVE,
            "label_horizon_bars": policy.get("label_horizon_bars"),
            "min_mfe_return": policy.get("min_mfe_return"),
            "min_reward_risk_ratio": policy.get("min_reward_risk_ratio"),
            "max_adverse_return": policy.get("max_adverse_return"),
        }
    if objective == TRADE_PATH_LABEL_OBJECTIVE:
        return {
            "label_objective": TRADE_PATH_LABEL_OBJECTIVE,
            "teacher_param_policy": policy.get("teacher_param_policy"),
            "event_scope": policy.get("event_scope"),
            "feature_snapshot": policy.get("feature_snapshot"),
            "initial_miss_buy_status": policy.get("initial_miss_buy_status"),
            "continuation_event_identity": policy.get("continuation_event_identity"),
            "label_contract_version": policy.get("label_contract_version"),
            "label_status_values": policy.get("label_status_values"),
            "filled_positive_rule": policy.get("filled_positive_rule"),
            "filled_nonpositive_rule": policy.get("filled_nonpositive_rule"),
            "filled_data_end_rule": policy.get("filled_data_end_rule"),
            "sizing_capital_rule": policy.get("sizing_capital_rule"),
            "unfilled_terminal_rule": policy.get("unfilled_terminal_rule"),
        }
    raise ValueError(f"不支援的 breakout quality label_objective: {objective or 'missing'}")


DEFAULT_LABEL_POLICY = BreakoutQualityLabelPolicy()


def trade_path_label_policy_payload() -> dict:
    payload = DEFAULT_LABEL_POLICY.as_manifest_payload()
    payload.update(
        {
            "label_objective": TRADE_PATH_LABEL_OBJECTIVE,
            "teacher_param_policy": "a2_dl_off_trained_point_in_time",
            "event_scope": "original_breakout_lifecycle",
            "feature_snapshot": "original_signal_date",
            "initial_miss_buy_status": "pending",
            "continuation_event_identity": "reuse_original_event",
            "label_contract_version": TRADE_PATH_LABEL_CONTRACT_VERSION,
            "label_status_values": {
                TRADE_PATH_LABEL_STATUS_PASS: 1,
                TRADE_PATH_LABEL_STATUS_REJECT: 0,
                TRADE_PATH_LABEL_STATUS_EXCLUDED: -1,
            },
            "filled_positive_rule": "realized_net_r_gt_zero",
            "filled_nonpositive_rule": "realized_net_r_le_zero",
            "filled_data_end_rule": "formal_single_stock_forced_closeout",
            "sizing_capital_rule": "same_explicit_single_stock_sizing_capital",
            "unfilled_terminal_rule": "exclude_from_binary_training",
        }
    )
    return payload


def expected_label_policy_for_filter_id(filter_id: str) -> dict:
    if str(filter_id).strip() == TRADE_PATH_FILTER_ID:
        return trade_path_label_policy_payload()
    return DEFAULT_LABEL_POLICY.as_manifest_payload()

LABEL_REJECT = 0
LABEL_PASS = 1
LABEL_INVALID = -1  # Internal sentinel only; it is not a third training label.
LABEL_NAME_MAP = {
    LABEL_INVALID: "INVALID",
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
    "DEFAULT_EXPERIMENT_PROFILE",
    "DEFAULT_EXPERIMENT_SETTINGS",
    "DEFAULT_MODEL_ARCHITECTURE",
    "DEFAULT_SCORE_FILENAME",
    "DEFAULT_UNAVAILABLE_SCORE_FILENAME",
    "DEFAULT_SPLIT_FILENAME",
    "FEATURE_COLUMNS",
    "FILTER_FAMILY",
    "LABEL_INVALID",
    "LABEL_OBJECTIVE",
    "TRADE_PATH_LABEL_OBJECTIVE",
    "TRADE_PATH_FILTER_ID",
    "TRADE_PATH_LABEL_CONTRACT_VERSION",
    "TRADE_PATH_LABEL_STATUS_EXCLUDED",
    "TRADE_PATH_LABEL_STATUS_PASS",
    "TRADE_PATH_LABEL_STATUS_REJECT",
    "LEGACY_LABEL_OBJECTIVE",
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
    "RUNTIME_SCOPE_WORKFLOW",
    "RUNTIME_SCOPE_NOT_EXPORTED",
    "RUNTIME_SCOPE_RESEARCH",
    "RUNTIME_SCOPE_ROLLING_OOS",
    "SCORE_TABLE_REQUIRED_COLUMNS",
    "SELECTION_ROLE_EMBARGO",
    "SELECTION_ROLE_INVALID",
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
    "expected_label_policy_for_filter_id",
    "label_manifest_payload_from_policy_manifest",
    "trade_path_label_policy_payload",
]
