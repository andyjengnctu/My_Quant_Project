"""Canonical breakout-quality architecture descriptor registry.

Architecture identity, active eligibility, spec-builder routing, runtime-builder
routing and architecture-only capability tags live here.  Generic consumers may
resolve capabilities/keys from the descriptor but must not maintain their own
architecture-ID membership switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TINY_CNN_V1 = "tiny_cnn_v1"
MULTISCALE_CNN_V1 = "multiscale_cnn_v1"
MULTISCALE_CNN_V2 = "multiscale_cnn_v2"
MULTISCALE_CNN_V3 = "multiscale_cnn_v3"
MULTISCALE_CNN_V4 = "multiscale_cnn_v4"
MULTISCALE_CNN_V5 = "multiscale_cnn_v5"
MULTISCALE_CNN_V6 = "multiscale_cnn_v6"
MULTISCALE_CNN_V7 = "multiscale_cnn_v7"
MULTISCALE_CNN_V8 = "multiscale_cnn_v8"
MULTISCALE_CNN_REGIME_CONTEXT_V1 = "multiscale_cnn_regime_context_v1"
MULTISCALE_CNN_SEQUENCE_ONLY_V1 = "multiscale_cnn_sequence_only_v1"
MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1 = "multiscale_cnn_sequence_only_dual_path_v1"
INCEPTION_TIME_V1 = "inception_time_v1"
INCEPTION_TIME_STOCK_EMA_V1 = "inception_time_stock_ema_v1"
INCEPTION_TIME_TRADE_DECISION_CONTEXT_V1 = "inception_time_trade_decision_context_v1"
INCEPTION_TIME_JOINT_QUADRANT_V1 = "inception_time_joint_quadrant_v1"
INCEPTION_TIME_RISK_CONTEXT_V1 = "inception_time_risk_context_v1"
INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1 = "inception_time_predicted_upside_context_v1"
INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1 = "inception_time_predicted_safety_context_v1"
INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1 = "inception_time_conditional_mfe_safety_v1"
INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1 = "inception_time_safety_conditional_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_V1 = "inception_time_shared_safety_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_ADVERSE_QUANTILE_AUX_V1 = (
    "inception_time_shared_safety_mfe_adverse_quantile_aux_v1"
)
INCEPTION_TIME_SHARED_SAFETY_MFE_TRADE_COUNT_REAL_V1 = "inception_time_shared_safety_mfe_pit_trade_count_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_TRADE_COUNT_PLACEBO_V1 = "inception_time_shared_safety_mfe_pit_trade_count_placebo_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_RAW_INPUT_V1 = "inception_time_shared_safety_mfe_raw_input_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_RAW_MASK_V1 = "inception_time_shared_safety_mfe_raw_quote_mask_v1"
INCEPTION_TIME_SHARED_SAFETY_DYNAMIC_HYPERGRAPH_MFE_V1 = (
    "inception_time_shared_safety_dynamic_hypergraph_mfe_v1"
)
INCEPTION_TIME_SHARED_SAFETY_DYNAMIC_HYPERGRAPH_RELATION_CHANGE_MFE_V1 = (
    "inception_time_shared_safety_dynamic_hypergraph_relation_change_mfe_v1"
)
INCEPTION_TIME_SHARED_ADAPTIVE_HORIZON_SAFETY_MFE_V1 = (
    "inception_time_shared_adaptive_horizon_safety_mfe_v1"
)
INCEPTION_TIME_SHARED_ADAPTIVE_INPUT_CONTEXT_SAFETY_MFE_V1 = (
    "inception_time_shared_adaptive_input_context_safety_mfe_v1"
)
INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_STRUCTURE_V1 = (
    "inception_time_shared_safety_mfe_price_volume_structure_v1"
)
INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_STRUCTURE_LOCAL_V1 = (
    "inception_time_shared_safety_mfe_price_volume_structure_local_v1"
)
INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_MULTISCALE_V1 = (
    "inception_time_shared_safety_mfe_price_volume_multiscale_v1"
)
INCEPTION_TIME_SHARED_SAFETY_MFE_PRICE_VOLUME_POSITION_AWARE_MULTISCALE_V1 = (
    "inception_time_shared_safety_mfe_price_volume_position_aware_multiscale_v1"
)
INCEPTION_TIME_SHARED_SAFETY_PAIRWISE_RELATION_MFE_V1 = (
    "inception_time_shared_safety_pairwise_relation_mfe_v1"
)
INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1 = "inception_time_task_specific_safety_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1 = "inception_time_shared_safety_attn_mfe_v1"
INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1 = "inception_time_task_specific_safety_attn_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1 = "inception_time_shared_safety_self_attn_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1 = "inception_time_shared_safety_mfe_full_window_rf_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1 = "inception_time_shared_safety_mfe_wide_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_V1 = "inception_time_shared_safety_mfe_600bar_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_DEEP_V1 = "inception_time_shared_safety_mfe_deep_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_WIDE_V1 = "inception_time_shared_safety_mfe_600bar_wide_v1"
DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_MFE_V1 = "day_token_transformer_shared_safety_mfe_v1"
GRU_SHARED_SAFETY_MFE_V1 = "gru_shared_safety_mfe_v1"
GRU_SHARED_SAFETY_MFE_V2 = "gru_shared_safety_mfe_v2"
GRU_SHARED_SAFETY_MFE_V3 = "gru_shared_safety_mfe_v3"
GRU_SHARED_SAFETY_MFE_V4 = "gru_shared_safety_mfe_v4"
BIGRU_RANKER_V1 = "bigru_ranker_v1"
PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1 = "patch_transformer_safety_inception_mfe_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1 = "inception_time_safety_raw_mfe_hmhs_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1 = "inception_time_safety_raw_mfe_hmhs_mlp_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = "inception_time_safety_raw_mfe_joint_attn_mlp_v1"
INCEPTION_TIME_MARKET_SET_V1 = "inception_time_market_set_v1"
INCEPTION_TIME_MARKET_SET_CANDIDATE_V1 = "inception_time_market_set_candidate_v1"
INCEPTION_TIME_GROUP_NORM_V1 = "inception_time_group_norm_v1"
MODERN_TCN_V1 = "modern_tcn_v1"
MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = "modern_tcn_safety_raw_mfe_joint_attn_mlp_v1"
PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1 = (
    "patch_token_transformer_safety_raw_mfe_joint_attn_mlp_v1"
)
PATCH_TOKEN_TRANSFORMER_RANKER_V1 = "patch_token_transformer_ranker_v1"
MANTIS_V2_FROZEN_LINEAR_V1 = "mantis_v2_frozen_linear_v1"
MOMENT_1_BASE_FROZEN_LINEAR_V1 = "moment_1_base_frozen_linear_v1"
PATCH_TRANSFORMER_V1 = "patch_transformer_v1"
TS2VEC_FROZEN_LINEAR_V1 = "ts2vec_frozen_linear_v1"
RESIDUAL_TCN_V1 = "residual_tcn_v1"


@dataclass(frozen=True)
class ArchitectureDescriptor:
    architecture_id: str
    active: bool
    spec_builder_key: str
    runtime_builder_key: str
    active_order: int | None = None
    capabilities: tuple[str, ...] = ()
    spec_options: tuple[tuple[str, Any], ...] = ()

    def has_capability(self, capability: str) -> bool:
        return str(capability) in self.capabilities

    def spec_options_dict(self) -> dict[str, Any]:
        return dict(self.spec_options)


def _descriptor(
    architecture_id: str,
    *,
    active: bool,
    spec_builder: str,
    runtime_builder: str,
    active_order: int | None = None,
    capabilities: tuple[str, ...] = (),
    **spec_options: Any,
) -> ArchitectureDescriptor:
    return ArchitectureDescriptor(
        architecture_id=architecture_id,
        active=bool(active),
        spec_builder_key=str(spec_builder),
        runtime_builder_key=str(runtime_builder),
        active_order=None if active_order is None else int(active_order),
        capabilities=tuple(str(value) for value in capabilities),
        spec_options=tuple(spec_options.items()),
    )


_ARCHITECTURE_DESCRIPTORS = (
    _descriptor(INCEPTION_TIME_V1, active=True, active_order=0, spec_builder="inception_variant", runtime_builder="inception_time", family="inception_time", pooling=("global_average",), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_STOCK_EMA_V1, active=True, active_order=1, spec_builder="inception_variant", runtime_builder="inception_time", family="inception_time", pooling=("global_average",), use_dataset_context=False, sequence_input_paths=("raw_level", "full_history_stock_close_ema"), stock_close_ema_spans=(5, 20, 60), head_width=None),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_V1, active=True, active_order=6, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe",), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
)

ARCHITECTURE_DESCRIPTORS: dict[str, ArchitectureDescriptor] = {
    descriptor.architecture_id: descriptor for descriptor in _ARCHITECTURE_DESCRIPTORS
}
if len(ARCHITECTURE_DESCRIPTORS) != len(_ARCHITECTURE_DESCRIPTORS):
    raise RuntimeError("breakout-quality architecture descriptor identity 重複")

SUPPORTED_MODEL_ARCHITECTURES = tuple(ARCHITECTURE_DESCRIPTORS)
ACTIVE_MODEL_ARCHITECTURES = tuple(
    descriptor.architecture_id
    for descriptor in sorted(
        (value for value in ARCHITECTURE_DESCRIPTORS.values() if value.active),
        key=lambda value: int(value.active_order if value.active_order is not None else 10**9),
    )
)
LEGACY_MODEL_ARCHITECTURES = tuple(
    architecture_id
    for architecture_id, descriptor in ARCHITECTURE_DESCRIPTORS.items()
    if not descriptor.active
)


def normalize_model_architecture(value: str) -> str:
    architecture = str(value).strip().lower()
    if architecture not in ARCHITECTURE_DESCRIPTORS:
        allowed = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
        raise ValueError(
            f"不支援的 breakout quality model architecture: {value!r}；可用值: {allowed}"
        )
    return architecture


def get_architecture_descriptor(value: str) -> ArchitectureDescriptor:
    return ARCHITECTURE_DESCRIPTORS[normalize_model_architecture(value)]


def architecture_has_capability(value: str, capability: str) -> bool:
    return get_architecture_descriptor(value).has_capability(capability)


def normalize_active_model_architecture(value: str) -> str:
    architecture = normalize_model_architecture(value)
    if not ARCHITECTURE_DESCRIPTORS[architecture].active:
        allowed = ", ".join(ACTIVE_MODEL_ARCHITECTURES)
        raise ValueError(
            "breakout quality 正式新訓練只允許 active architecture；"
            f"收到 {architecture!r}，可用值: {allowed}。歷史版本僅供舊工件重現。"
        )
    return architecture
