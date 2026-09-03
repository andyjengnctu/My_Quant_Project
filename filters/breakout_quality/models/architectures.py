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
INCEPTION_TIME_RISK_CONTEXT_V1 = "inception_time_risk_context_v1"
INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1 = "inception_time_predicted_upside_context_v1"
INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1 = "inception_time_predicted_safety_context_v1"
INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1 = "inception_time_conditional_mfe_safety_v1"
INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1 = "inception_time_safety_conditional_mfe_v1"
INCEPTION_TIME_SHARED_SAFETY_MFE_V1 = "inception_time_shared_safety_mfe_v1"
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
    _descriptor(TINY_CNN_V1, active=False, spec_builder="tiny_cnn", runtime_builder="tiny_cnn"),
    _descriptor(MULTISCALE_CNN_V1, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn"),
    _descriptor(MULTISCALE_CNN_V2, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_input_representations=("return_delta", "return_delta", "level")),
    _descriptor(MULTISCALE_CNN_V3, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_input_representations=("market_relative_return_delta", "market_relative_return_delta", "level")),
    _descriptor(MULTISCALE_CNN_V4, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_channels=(16, 16, 8)),
    _descriptor(MULTISCALE_CNN_V5, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_channels=(16, 16, 12)),
    _descriptor(MULTISCALE_CNN_V6, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_dropouts=(0.25, 0.25, 0.40)),
    _descriptor(MULTISCALE_CNN_V7, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_input_representations=("return_delta", "level", "level")),
    _descriptor(MULTISCALE_CNN_V8, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", branch_input_representations=("level", "return_delta", "level")),
    _descriptor(MULTISCALE_CNN_REGIME_CONTEXT_V1, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", capabilities=("regime_context",)),
    _descriptor(MULTISCALE_CNN_SEQUENCE_ONLY_V1, active=True, active_order=14, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", use_dataset_context=False),
    _descriptor(MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1, active=False, spec_builder="multiscale_cnn", runtime_builder="multiscale_cnn", use_dataset_context=False, sequence_input_paths=("raw_level", "window_zscore"), window_normalization_epsilon=1e-5),
    _descriptor(INCEPTION_TIME_V1, active=True, active_order=0, spec_builder="inception_variant", runtime_builder="inception_time", family="inception_time", pooling=("global_average",), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_RISK_CONTEXT_V1, active=True, active_order=1, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("risk_context",), family="inception_time_risk_context", pooling=("global_average", "risk_context_mlp_concat"), use_dataset_context=True, sequence_input_paths=("raw_level", "universal_risk_economic_context"), head_width=16),
    _descriptor(INCEPTION_TIME_PREDICTED_UPSIDE_CONTEXT_V1, active=True, active_order=2, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("predicted_upside_context",), family="inception_time_predicted_upside_context", pooling=("global_average", "predicted_upside_percentile_concat"), use_dataset_context=True, sequence_input_paths=("raw_level", "pit_safe_predicted_upside_percentile"), head_width=None),
    _descriptor(INCEPTION_TIME_PREDICTED_SAFETY_CONTEXT_V1, active=True, active_order=3, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("predicted_safety_context",), family="inception_time_predicted_safety_context", pooling=("global_average", "predicted_safety_percentile_concat"), use_dataset_context=True, sequence_input_paths=("raw_level", "pit_safe_predicted_safety_percentile"), head_width=None),
    _descriptor(INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1, active=True, active_order=4, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("conditional_mfe_safety",), family="inception_time_conditional_mfe_safety", pooling=("global_average", "primary_mfe_head", "conditional_safety_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1, active=True, active_order=5, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("safety_conditional_mfe",), family="inception_time_safety_conditional_mfe", pooling=("global_average", "raw_safety_head", "conditional_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_V1, active=True, active_order=6, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe",), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_TASK_SPECIFIC_SAFETY_MFE_V1, active=True, active_order=7, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("task_specific_safety_mfe",), family="inception_time_task_specific_safety_mfe", pooling=("task_specific_final_residual_group", "global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_ATTN_MFE_V1, active=True, active_order=15, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "safety_attention_pool"), family="inception_time_shared_safety_attn_mfe", pooling=("safety_scalar_attention_pool", "mfe_global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_TASK_SPECIFIC_SAFETY_ATTN_MFE_V1, active=True, active_order=16, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("task_specific_safety_mfe", "safety_attention_pool"), family="inception_time_task_specific_safety_attn_mfe", pooling=("task_specific_final_residual_group", "safety_scalar_attention_pool", "mfe_global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_SELF_ATTN_MFE_V1, active=True, active_order=17, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "safety_temporal_self_attention"), family="inception_time_shared_safety_self_attn_mfe", pooling=("safety_single_head_temporal_self_attention_residual", "safety_global_average", "mfe_global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_FULL_WINDOW_RF_V1, active=True, active_order=18, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "full_window_receptive_field"), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None, inception_module_dilations=(1, 1, 1, 1, 2, 2)),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_WIDE_V1, active=True, active_order=20, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "wide_capacity"), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None, inception_filters=64, inception_bottleneck_channels=64),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_V1, active=True, active_order=21, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "long_horizon_input"), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None, input_window_bars=600, inception_module_dilations=(1, 1, 1, 1, 6, 6)),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_DEEP_V1, active=True, active_order=22, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "deep_hierarchy_capacity_matched"), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None, inception_depth=12, inception_filters=32, inception_bottleneck_channels=24, inception_kernel_sizes=(21, 11, 5), inception_residual_every=3),
    _descriptor(INCEPTION_TIME_SHARED_SAFETY_MFE_600BAR_WIDE_V1, active=True, active_order=23, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("shared_safety_mfe", "long_horizon_input", "wide_capacity", "long_horizon_capacity_interaction"), family="inception_time_shared_safety_mfe", pooling=("global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None, input_window_bars=600, inception_filters=64, inception_bottleneck_channels=64, inception_module_dilations=(1, 1, 1, 1, 6, 6)),
    _descriptor(DAY_TOKEN_TRANSFORMER_SHARED_SAFETY_MFE_V1, active=True, active_order=24, spec_builder="day_token_transformer_shared_safety_mfe", runtime_builder="day_token_transformer_shared_safety_mfe", capabilities=("shared_safety_mfe", "global_day_token_attention")),
    _descriptor(PATCH_TRANSFORMER_SAFETY_INCEPTION_MFE_V1, active=True, active_order=19, spec_builder="hybrid_safety_patch_mfe_inception", runtime_builder="hybrid_safety_patch_mfe_inception", capabilities=("independent_safety_patch_mfe_inception",), family="patch_transformer_safety_inception_mfe", pooling=("safety_patch_token_global_average", "mfe_inception_global_average", "raw_safety_head", "raw_mfe_head"), use_dataset_context=False, sequence_input_paths=("raw_level", "raw_level_temporal_nonoverlap_patches"), head_width=None),
    _descriptor(INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1, active=True, active_order=8, spec_builder="inception_variant", runtime_builder="inception_time", capabilities=("safety_raw_mfe_hmhs",), family="inception_time_safety_raw_mfe_hmhs", pooling=("global_average", "raw_safety_head", "safety_conditioned_raw_mfe_head", "direct_hmhs_head"), use_dataset_context=False, sequence_input_paths=("raw_level",), head_width=None),
    _descriptor(INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1, active=True, active_order=9, spec_builder="inception_hmhs_mlp", runtime_builder="inception_time", capabilities=("safety_raw_mfe_hmhs", "nonlinear_hmhs_head")),
    _descriptor(INCEPTION_TIME_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1, active=True, active_order=10, spec_builder="inception_joint", runtime_builder="inception_time", capabilities=("safety_raw_mfe_hmhs", "nonlinear_hmhs_head", "joint_attention_pool")),
    _descriptor(INCEPTION_TIME_MARKET_SET_V1, active=False, spec_builder="market_set", runtime_builder="market_set"),
    _descriptor(INCEPTION_TIME_MARKET_SET_CANDIDATE_V1, active=False, spec_builder="market_set", runtime_builder="market_set", capabilities=("candidate_conditioned_market_set",)),
    _descriptor(INCEPTION_TIME_GROUP_NORM_V1, active=False, spec_builder="inception_group_norm", runtime_builder="inception_time"),
    _descriptor(MODERN_TCN_V1, active=False, spec_builder="modern_tcn", runtime_builder="modern_tcn"),
    _descriptor(MODERN_TCN_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1, active=True, active_order=11, spec_builder="modern_tcn_joint", runtime_builder="modern_tcn_joint"),
    _descriptor(PATCH_TOKEN_TRANSFORMER_SAFETY_RAW_MFE_JOINT_ATTN_MLP_V1, active=True, active_order=12, spec_builder="patch_token_joint", runtime_builder="patch_token_joint"),
    _descriptor(PATCH_TOKEN_TRANSFORMER_RANKER_V1, active=True, active_order=13, spec_builder="patch_token_ranker", runtime_builder="patch_token_ranker"),
    _descriptor(MANTIS_V2_FROZEN_LINEAR_V1, active=False, spec_builder="mantis", runtime_builder="mantis"),
    _descriptor(MOMENT_1_BASE_FROZEN_LINEAR_V1, active=False, spec_builder="moment", runtime_builder="moment"),
    _descriptor(PATCH_TRANSFORMER_V1, active=False, spec_builder="patch_transformer", runtime_builder="patch_transformer"),
    _descriptor(TS2VEC_FROZEN_LINEAR_V1, active=False, spec_builder="ts2vec", runtime_builder="ts2vec"),
    _descriptor(RESIDUAL_TCN_V1, active=False, spec_builder="residual_tcn", runtime_builder="residual_tcn"),
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
