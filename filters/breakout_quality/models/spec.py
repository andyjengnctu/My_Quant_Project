"""Versioned model specifications for breakout quality classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from config.breakout_quality import (
    BREAKOUT_QUALITY_INCEPTION_DEPTH,
    BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY,
    BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS,
    BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES,
    BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT,
    BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM,
    BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM,
    BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS,
    BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS,
    BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH,
    BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO,
    BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT,
    BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS,
    BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE,
    build_breakout_quality_inception_kernel_sizes,
    resolve_breakout_quality_inception_receptive_field_bars,
)

from filters.breakout_quality.models.regime_context import (
    REGIME_CONTEXT_ANNUALIZATION_BARS,
    REGIME_CONTEXT_FEATURES,
    REGIME_CONTEXT_LOOKBACK_BARS,
)


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
INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1 = "inception_time_conditional_mfe_safety_v1"
INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1 = "inception_time_safety_conditional_mfe_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1 = "inception_time_safety_raw_mfe_hmhs_v1"
INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1 = "inception_time_safety_raw_mfe_hmhs_mlp_v1"
INCEPTION_TIME_MARKET_SET_V1 = "inception_time_market_set_v1"
INCEPTION_TIME_MARKET_SET_CANDIDATE_V1 = "inception_time_market_set_candidate_v1"
INCEPTION_TIME_GROUP_NORM_V1 = "inception_time_group_norm_v1"
MODERN_TCN_V1 = "modern_tcn_v1"
MANTIS_V2_FROZEN_LINEAR_V1 = "mantis_v2_frozen_linear_v1"
MOMENT_1_BASE_FROZEN_LINEAR_V1 = "moment_1_base_frozen_linear_v1"
PATCH_TRANSFORMER_V1 = "patch_transformer_v1"
TS2VEC_FROZEN_LINEAR_V1 = "ts2vec_frozen_linear_v1"
RESIDUAL_TCN_V1 = "residual_tcn_v1"
SUPPORTED_MODEL_ARCHITECTURES = (
    TINY_CNN_V1,
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_V2,
    MULTISCALE_CNN_V3,
    MULTISCALE_CNN_V4,
    MULTISCALE_CNN_V5,
    MULTISCALE_CNN_V6,
    MULTISCALE_CNN_V7,
    MULTISCALE_CNN_V8,
    MULTISCALE_CNN_REGIME_CONTEXT_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    INCEPTION_TIME_V1,
    INCEPTION_TIME_RISK_CONTEXT_V1,
    INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
    INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
    INCEPTION_TIME_MARKET_SET_V1,
    INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    INCEPTION_TIME_GROUP_NORM_V1,
    MODERN_TCN_V1,
    MANTIS_V2_FROZEN_LINEAR_V1,
    MOMENT_1_BASE_FROZEN_LINEAR_V1,
    PATCH_TRANSFORMER_V1,
    TS2VEC_FROZEN_LINEAR_V1,
    RESIDUAL_TCN_V1,
)
ACTIVE_MODEL_ARCHITECTURES = (
    INCEPTION_TIME_V1,
    INCEPTION_TIME_RISK_CONTEXT_V1,
    INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1,
    INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1,
    INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1,
    MULTISCALE_CNN_SEQUENCE_ONLY_V1,
)
LEGACY_MODEL_ARCHITECTURES = tuple(
    architecture
    for architecture in SUPPORTED_MODEL_ARCHITECTURES
    if architecture not in ACTIVE_MODEL_ARCHITECTURES
)


@dataclass(frozen=True)
class BreakoutQualityModelSpec:
    architecture: str
    family: str
    channels: int
    kernel_size: int
    dilations: tuple[int, ...]
    convolutions_per_block: int
    pooling: tuple[str, ...]
    dropout: float
    receptive_field_bars: int
    normalization: str | None = None
    normalization_groups: int | None = None
    head_width: int | None = None
    branch_downsample_factors: tuple[int, ...] = ()
    branch_kernel_sizes: tuple[tuple[int, ...], ...] = ()
    branch_summary_windows_bars: tuple[tuple[int, ...], ...] = ()
    branch_input_representations: tuple[str, ...] = ()
    branch_channels: tuple[int, ...] = ()
    branch_dropouts: tuple[float, ...] = ()
    use_dataset_context: bool = True
    derived_context_features: tuple[str, ...] = ()
    derived_context_lookback_bars: tuple[int, ...] = ()
    derived_context_annualization_bars: int | None = None
    sequence_input_paths: tuple[str, ...] = ()
    window_normalization_epsilon: float | None = None
    inception_depth: int | None = None
    inception_filters: int | None = None
    inception_bottleneck_channels: int | None = None
    inception_kernel_sizes: tuple[int, ...] = ()
    inception_residual_every: int | None = None
    requires_market_set: bool = False
    market_set_history_bars: int | None = None
    market_set_base_features: tuple[str, ...] = ()
    market_set_stock_embedding_dim: int | None = None
    market_set_temporal_channels: int | None = None
    market_set_temporal_kernel_size: int | None = None
    market_set_temporal_stride: int | None = None
    market_set_temporal_dilations: tuple[int, ...] = ()
    market_set_temporal_normalization: str | None = None
    market_set_temporal_normalization_groups: int | None = None
    market_set_query_mode: str | None = None
    market_set_query_count: int | None = None
    market_set_attention_heads: int | None = None
    market_set_embedding_dim: int | None = None
    market_set_fusion_hidden_dim: int | None = None
    market_set_min_valid_history_ratio: float | None = None
    market_set_max_stocks: int | None = None
    market_set_max_dates_per_batch: int | None = None
    modern_tcn_depth: int | None = None
    modern_tcn_channels: int | None = None
    modern_tcn_kernel_size: int | None = None
    modern_tcn_expansion_ratio: int | None = None
    ts2vec_hidden_dims: int | None = None
    ts2vec_output_dims: int | None = None
    ts2vec_depth: int | None = None
    ts2vec_temporal_unit: int | None = None
    mantis_repository: str | None = None
    mantis_revision: str | None = None
    mantis_input_length: int | None = None
    mantis_num_patches: int | None = None
    mantis_hidden_dim: int | None = None
    mantis_embedding_dim: int | None = None
    mantis_scalar_hidden_dim: int | None = None
    mantis_scalar_epsilon: float | None = None
    mantis_transformer_depth: int | None = None
    mantis_transformer_heads: int | None = None
    mantis_transformer_mlp_dim: int | None = None
    mantis_transformer_dim_head: int | None = None
    mantis_return_transformer_layer: int | None = None
    mantis_output_token: str | None = None
    mantis_channel_aggregation: str | None = None
    moment_repository: str | None = None
    moment_revision: str | None = None
    moment_input_length: int | None = None
    moment_patch_length: int | None = None
    moment_patch_stride: int | None = None
    moment_embedding_dim: int | None = None
    moment_transformer_layers: int | None = None
    moment_transformer_heads: int | None = None
    moment_channel_aggregation: str | None = None
    moment_patch_reduction: str | None = None
    patch_transformer_patch_size: int | None = None
    patch_transformer_patch_stride: int | None = None
    patch_transformer_embedding_dim: int | None = None
    patch_transformer_depth: int | None = None
    patch_transformer_heads: int | None = None
    patch_transformer_mlp_dim: int | None = None
    patch_transformer_pooling: str | None = None
    patch_transformer_positional_encoding: str | None = None

    def as_manifest_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "architecture": self.architecture,
            "family": self.family,
            "channels": int(self.channels),
            "kernel_size": int(self.kernel_size),
            "dilations": list(self.dilations),
            "convolutions_per_block": int(self.convolutions_per_block),
            "pooling": list(self.pooling),
            "dropout": float(self.dropout),
            "receptive_field_bars": int(self.receptive_field_bars),
        }
        optional_scalars = {
            "normalization": self.normalization,
            "normalization_groups": self.normalization_groups,
            "head_width": self.head_width,
            "inception_depth": self.inception_depth,
            "inception_filters": self.inception_filters,
            "inception_bottleneck_channels": self.inception_bottleneck_channels,
            "inception_residual_every": self.inception_residual_every,
            "market_set_history_bars": self.market_set_history_bars,
            "market_set_stock_embedding_dim": self.market_set_stock_embedding_dim,
            "market_set_temporal_channels": self.market_set_temporal_channels,
            "market_set_temporal_kernel_size": self.market_set_temporal_kernel_size,
            "market_set_temporal_stride": self.market_set_temporal_stride,
            "market_set_temporal_normalization": self.market_set_temporal_normalization,
            "market_set_temporal_normalization_groups": self.market_set_temporal_normalization_groups,
            "market_set_query_mode": self.market_set_query_mode,
            "market_set_query_count": self.market_set_query_count,
            "market_set_attention_heads": self.market_set_attention_heads,
            "market_set_embedding_dim": self.market_set_embedding_dim,
            "market_set_fusion_hidden_dim": self.market_set_fusion_hidden_dim,
            "market_set_min_valid_history_ratio": self.market_set_min_valid_history_ratio,
            "market_set_max_stocks": self.market_set_max_stocks,
            "market_set_max_dates_per_batch": self.market_set_max_dates_per_batch,
            "modern_tcn_depth": self.modern_tcn_depth,
            "modern_tcn_channels": self.modern_tcn_channels,
            "modern_tcn_kernel_size": self.modern_tcn_kernel_size,
            "modern_tcn_expansion_ratio": self.modern_tcn_expansion_ratio,
            "ts2vec_hidden_dims": self.ts2vec_hidden_dims,
            "ts2vec_output_dims": self.ts2vec_output_dims,
            "ts2vec_depth": self.ts2vec_depth,
            "ts2vec_temporal_unit": self.ts2vec_temporal_unit,
            "mantis_repository": self.mantis_repository,
            "mantis_revision": self.mantis_revision,
            "mantis_input_length": self.mantis_input_length,
            "mantis_num_patches": self.mantis_num_patches,
            "mantis_hidden_dim": self.mantis_hidden_dim,
            "mantis_embedding_dim": self.mantis_embedding_dim,
            "mantis_scalar_hidden_dim": self.mantis_scalar_hidden_dim,
            "mantis_scalar_epsilon": self.mantis_scalar_epsilon,
            "mantis_transformer_depth": self.mantis_transformer_depth,
            "mantis_transformer_heads": self.mantis_transformer_heads,
            "mantis_transformer_mlp_dim": self.mantis_transformer_mlp_dim,
            "mantis_transformer_dim_head": self.mantis_transformer_dim_head,
            "mantis_return_transformer_layer": self.mantis_return_transformer_layer,
            "mantis_output_token": self.mantis_output_token,
            "mantis_channel_aggregation": self.mantis_channel_aggregation,
            "moment_repository": self.moment_repository,
            "moment_revision": self.moment_revision,
            "moment_input_length": self.moment_input_length,
            "moment_patch_length": self.moment_patch_length,
            "moment_patch_stride": self.moment_patch_stride,
            "moment_embedding_dim": self.moment_embedding_dim,
            "moment_transformer_layers": self.moment_transformer_layers,
            "moment_transformer_heads": self.moment_transformer_heads,
            "moment_channel_aggregation": self.moment_channel_aggregation,
            "moment_patch_reduction": self.moment_patch_reduction,
            "patch_transformer_patch_size": self.patch_transformer_patch_size,
            "patch_transformer_patch_stride": self.patch_transformer_patch_stride,
            "patch_transformer_embedding_dim": self.patch_transformer_embedding_dim,
            "patch_transformer_depth": self.patch_transformer_depth,
            "patch_transformer_heads": self.patch_transformer_heads,
            "patch_transformer_mlp_dim": self.patch_transformer_mlp_dim,
            "patch_transformer_pooling": self.patch_transformer_pooling,
            "patch_transformer_positional_encoding": self.patch_transformer_positional_encoding,
        }
        for key, value in optional_scalars.items():
            if value is not None:
                payload[key] = value
        if self.branch_downsample_factors:
            payload["branch_downsample_factors"] = list(self.branch_downsample_factors)
        if self.branch_kernel_sizes:
            payload["branch_kernel_sizes"] = [list(values) for values in self.branch_kernel_sizes]
        if self.branch_summary_windows_bars:
            payload["branch_summary_windows_bars"] = [
                list(values) for values in self.branch_summary_windows_bars
            ]
        if self.branch_input_representations:
            payload["branch_input_representations"] = list(self.branch_input_representations)
        if self.branch_channels:
            payload["branch_channels"] = list(self.branch_channels)
        if self.branch_dropouts:
            payload["branch_dropouts"] = [float(value) for value in self.branch_dropouts]
        if not self.use_dataset_context:
            payload["use_dataset_context"] = False
        if self.requires_market_set:
            payload["requires_market_set"] = True
        if self.derived_context_features:
            payload["derived_context_features"] = list(self.derived_context_features)
        if self.derived_context_lookback_bars:
            payload["derived_context_lookback_bars"] = list(
                self.derived_context_lookback_bars
            )
        if self.derived_context_annualization_bars is not None:
            payload["derived_context_annualization_bars"] = int(
                self.derived_context_annualization_bars
            )
        if self.sequence_input_paths:
            payload["sequence_input_paths"] = list(self.sequence_input_paths)
        if self.window_normalization_epsilon is not None:
            payload["window_normalization_epsilon"] = float(
                self.window_normalization_epsilon
            )
        if self.inception_kernel_sizes:
            payload["inception_kernel_sizes"] = [
                int(value) for value in self.inception_kernel_sizes
            ]
        if self.market_set_base_features:
            payload["market_set_base_features"] = list(self.market_set_base_features)
        if self.market_set_temporal_dilations:
            payload["market_set_temporal_dilations"] = [
                int(value) for value in self.market_set_temporal_dilations
            ]
        return payload


def normalize_model_architecture(value: str) -> str:
    architecture = str(value).strip().lower()
    if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        allowed = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
        raise ValueError(
            f"不支援的 breakout quality model architecture: {value!r}；可用值: {allowed}"
        )
    return architecture


def normalize_active_model_architecture(value: str) -> str:
    architecture = normalize_model_architecture(value)
    if architecture not in ACTIVE_MODEL_ARCHITECTURES:
        allowed = ", ".join(ACTIVE_MODEL_ARCHITECTURES)
        raise ValueError(
            "breakout quality 正式新訓練只允許 active architecture；"
            f"收到 {architecture!r}，可用值: {allowed}。歷史版本僅供舊工件重現。"
        )
    return architecture


def _residual_receptive_field(
    *, kernel_size: int, dilations: tuple[int, ...], convolutions_per_block: int
) -> int:
    return 1 + int(convolutions_per_block) * (int(kernel_size) - 1) * sum(
        int(value) for value in dilations
    )


def _multiscale_receptive_field(
    *, downsample_factors: tuple[int, ...], kernel_sizes: tuple[tuple[int, ...], ...]
) -> int:
    if len(downsample_factors) != len(kernel_sizes):
        raise ValueError("multiscale branch spec 長度不一致")
    branch_fields = []
    for factor, kernels in zip(downsample_factors, kernel_sizes):
        if int(factor) < 1 or not kernels:
            raise ValueError("multiscale branch factor 與 kernels 必須有效")
        branch_fields.append(
            int(factor) * (1 + sum(int(kernel) - 1 for kernel in kernels))
        )
    return max(branch_fields)


def get_model_spec(architecture: str) -> BreakoutQualityModelSpec:
    normalized = normalize_model_architecture(architecture)
    if normalized == TINY_CNN_V1:
        return BreakoutQualityModelSpec(
            architecture=TINY_CNN_V1,
            family="tiny_cnn",
            channels=32,
            kernel_size=5,
            dilations=(1, 2),
            convolutions_per_block=1,
            pooling=("average",),
            dropout=0.10,
            receptive_field_bars=11,
        )

    if normalized in {
        MULTISCALE_CNN_V1,
        MULTISCALE_CNN_V2,
        MULTISCALE_CNN_V3,
        MULTISCALE_CNN_V4,
        MULTISCALE_CNN_V5,
        MULTISCALE_CNN_V6,
        MULTISCALE_CNN_V7,
        MULTISCALE_CNN_V8,
        MULTISCALE_CNN_REGIME_CONTEXT_V1,
        MULTISCALE_CNN_SEQUENCE_ONLY_V1,
        MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1,
    }:
        downsample_factors = (1, 2, 4)
        branch_kernel_sizes = ((3, 5), (9, 15), (31, 31))
        branch_summary_windows_bars = ((0, 20), (0, 60), (120, 300))
        branch_channels = ()
        branch_dropouts = ()
        use_dataset_context = True
        derived_context_features = ()
        derived_context_lookback_bars = ()
        derived_context_annualization_bars = None
        sequence_input_paths = ()
        window_normalization_epsilon = None
        if normalized == MULTISCALE_CNN_V1:
            branch_input_representations = ()
        elif normalized == MULTISCALE_CNN_V2:
            branch_input_representations = ("return_delta", "return_delta", "level")
        elif normalized == MULTISCALE_CNN_V3:
            branch_input_representations = (
                "market_relative_return_delta",
                "market_relative_return_delta",
                "level",
            )
        elif normalized == MULTISCALE_CNN_V4:
            branch_input_representations = ()
            branch_channels = (16, 16, 8)
        elif normalized == MULTISCALE_CNN_V5:
            branch_input_representations = ()
            branch_channels = (16, 16, 12)
        elif normalized == MULTISCALE_CNN_V6:
            branch_input_representations = ()
            branch_dropouts = (0.25, 0.25, 0.40)
        elif normalized == MULTISCALE_CNN_V7:
            branch_input_representations = ("return_delta", "level", "level")
        elif normalized == MULTISCALE_CNN_V8:
            branch_input_representations = ("level", "return_delta", "level")
        elif normalized == MULTISCALE_CNN_REGIME_CONTEXT_V1:
            branch_input_representations = ()
            derived_context_features = REGIME_CONTEXT_FEATURES
            derived_context_lookback_bars = REGIME_CONTEXT_LOOKBACK_BARS
            derived_context_annualization_bars = REGIME_CONTEXT_ANNUALIZATION_BARS
        elif normalized == MULTISCALE_CNN_SEQUENCE_ONLY_V1:
            branch_input_representations = ()
            use_dataset_context = False
        elif normalized == MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1:
            branch_input_representations = ()
            use_dataset_context = False
            sequence_input_paths = ("raw_level", "window_zscore")
            window_normalization_epsilon = 1e-5
        else:
            raise AssertionError(f"未處理的 multiscale architecture: {normalized}")
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="multiscale_cnn",
            channels=16,
            kernel_size=31,
            dilations=(),
            convolutions_per_block=2,
            pooling=("last", "window_average"),
            dropout=0.25,
            receptive_field_bars=_multiscale_receptive_field(
                downsample_factors=downsample_factors,
                kernel_sizes=branch_kernel_sizes,
            ),
            normalization="group_norm",
            normalization_groups=4,
            head_width=32,
            branch_downsample_factors=downsample_factors,
            branch_kernel_sizes=branch_kernel_sizes,
            branch_summary_windows_bars=branch_summary_windows_bars,
            branch_input_representations=branch_input_representations,
            branch_channels=branch_channels,
            branch_dropouts=branch_dropouts,
            use_dataset_context=use_dataset_context,
            derived_context_features=derived_context_features,
            derived_context_lookback_bars=derived_context_lookback_bars,
            derived_context_annualization_bars=derived_context_annualization_bars,
            sequence_input_paths=sequence_input_paths,
            window_normalization_epsilon=window_normalization_epsilon,
        )


    if normalized == PATCH_TRANSFORMER_V1:
        patch_size = 10
        embedding_dim = 128
        depth = 3
        heads = 4
        mlp_dim = 256
        return BreakoutQualityModelSpec(
            architecture=PATCH_TRANSFORMER_V1,
            family="patch_transformer",
            channels=embedding_dim,
            kernel_size=patch_size,
            dilations=(),
            convolutions_per_block=1,
            pooling=("patch_mean",),
            dropout=0.10,
            receptive_field_bars=300,
            normalization="layer_norm",
            use_dataset_context=False,
            sequence_input_paths=("raw_level_temporal_nonoverlap_patches",),
            patch_transformer_patch_size=patch_size,
            patch_transformer_patch_stride=patch_size,
            patch_transformer_embedding_dim=embedding_dim,
            patch_transformer_depth=depth,
            patch_transformer_heads=heads,
            patch_transformer_mlp_dim=mlp_dim,
            patch_transformer_pooling="mean",
            patch_transformer_positional_encoding="sinusoidal",
        )

    if normalized == MOMENT_1_BASE_FROZEN_LINEAR_V1:
        from filters.breakout_quality.moment_contract import (
            MOMENT_EMBEDDING_DIM,
            MOMENT_INPUT_LENGTH,
            MOMENT_PATCH_LENGTH,
            MOMENT_PATCH_STRIDE,
            MOMENT_REPOSITORY,
            MOMENT_REVISION,
            MOMENT_TRANSFORMER_HEADS,
            MOMENT_TRANSFORMER_LAYERS,
        )

        return BreakoutQualityModelSpec(
            architecture=MOMENT_1_BASE_FROZEN_LINEAR_V1,
            family="moment_frozen_linear",
            channels=MOMENT_EMBEDDING_DIM,
            kernel_size=MOMENT_PATCH_LENGTH,
            dilations=(),
            convolutions_per_block=1,
            pooling=("patch_mean", "channel_concat"),
            dropout=0.10,
            receptive_field_bars=300,
            normalization="official_moment_revin",
            use_dataset_context=False,
            sequence_input_paths=("raw_level_multichannel_linear_interpolate_512",),
            moment_repository=MOMENT_REPOSITORY,
            moment_revision=MOMENT_REVISION,
            moment_input_length=MOMENT_INPUT_LENGTH,
            moment_patch_length=MOMENT_PATCH_LENGTH,
            moment_patch_stride=MOMENT_PATCH_STRIDE,
            moment_embedding_dim=MOMENT_EMBEDDING_DIM,
            moment_transformer_layers=MOMENT_TRANSFORMER_LAYERS,
            moment_transformer_heads=MOMENT_TRANSFORMER_HEADS,
            moment_channel_aggregation="independent_channel_concat",
            moment_patch_reduction="mean",
        )

    if normalized == MANTIS_V2_FROZEN_LINEAR_V1:
        from filters.breakout_quality.mantis_contract import (
            MANTIS_V2_REPOSITORY,
            MANTIS_V2_REVISION,
        )

        return BreakoutQualityModelSpec(
            architecture=MANTIS_V2_FROZEN_LINEAR_V1,
            family="mantis_v2_frozen_linear",
            channels=256,
            kernel_size=41,
            dilations=(),
            convolutions_per_block=1,
            pooling=("transformer_layer_2_cls_mean_combined", "channel_concat"),
            dropout=0.10,
            receptive_field_bars=300,
            normalization="official_mantis_v2",
            use_dataset_context=False,
            sequence_input_paths=("raw_level_per_channel_linear_interpolate_512",),
            mantis_repository=MANTIS_V2_REPOSITORY,
            mantis_revision=MANTIS_V2_REVISION,
            mantis_input_length=512,
            mantis_num_patches=32,
            mantis_hidden_dim=256,
            mantis_embedding_dim=512,
            mantis_scalar_hidden_dim=32,
            mantis_scalar_epsilon=1.1,
            mantis_transformer_depth=6,
            mantis_transformer_heads=8,
            mantis_transformer_mlp_dim=512,
            mantis_transformer_dim_head=32,
            mantis_return_transformer_layer=2,
            mantis_output_token="combined",
            mantis_channel_aggregation="independent_channel_concat",
        )

    if normalized == TS2VEC_FROZEN_LINEAR_V1:
        depth = 8
        hidden_dims = 128
        output_dims = 320
        kernel_size = 3
        return BreakoutQualityModelSpec(
            architecture=TS2VEC_FROZEN_LINEAR_V1,
            family="ts2vec_frozen_linear",
            channels=hidden_dims,
            kernel_size=kernel_size,
            dilations=tuple(2**index for index in range(depth)),
            convolutions_per_block=2,
            pooling=("global_max",),
            dropout=0.10,
            receptive_field_bars=1 + 2 * (kernel_size - 1) * sum(2**index for index in range(depth)),
            normalization=None,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            ts2vec_hidden_dims=hidden_dims,
            ts2vec_output_dims=output_dims,
            ts2vec_depth=depth,
            ts2vec_temporal_unit=0,
        )

    if normalized == MODERN_TCN_V1:
        depth = 6
        channels = 96
        kernel_size = 51
        expansion_ratio = 4
        return BreakoutQualityModelSpec(
            architecture=MODERN_TCN_V1,
            family="modern_tcn",
            channels=channels,
            kernel_size=kernel_size,
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average",),
            dropout=0.10,
            receptive_field_bars=1 + depth * (kernel_size - 1),
            normalization="batch_norm",
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            modern_tcn_depth=depth,
            modern_tcn_channels=channels,
            modern_tcn_kernel_size=kernel_size,
            modern_tcn_expansion_ratio=expansion_ratio,
        )

    if normalized in {
        INCEPTION_TIME_MARKET_SET_V1,
        INCEPTION_TIME_MARKET_SET_CANDIDATE_V1,
    }:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        candidate_conditioned = normalized == INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
        query_count = int(
            BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT
            if candidate_conditioned
            else BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT
        )
        if int(BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS) < 2:
            raise ValueError("MARKET_SET_HISTORY_BARS 必須 >= 2")
        if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE) < 1:
            raise ValueError("MARKET_SET_TEMPORAL_KERNEL_SIZE 必須 >= 1")
        if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE) < 1:
            raise ValueError("MARKET_SET_TEMPORAL_STRIDE 必須 >= 1")
        if not BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS or any(
            int(value) < 1 for value in BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS
        ):
            raise ValueError("MARKET_SET_TEMPORAL_DILATIONS 必須是非空正整數")
        if query_count < 1:
            raise ValueError("market query count 必須 >= 1")
        if int(BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM) % int(
            BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS
        ) != 0:
            raise ValueError("market stock embedding dim 必須可被 attention heads 整除")
        if str(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION) != "group_norm":
            raise ValueError("market temporal normalization 第一版固定為 group_norm")
        if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS) % int(
            BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS
        ) != 0:
            raise ValueError("market temporal channels 必須可被 normalization groups 整除")
        if not 0.0 < float(BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO) <= 1.0:
            raise ValueError("market min valid history ratio 必須介於 0 與 1")
        if int(BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS) < 0:
            raise ValueError("market max stocks 必須 >= 0")
        if int(BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH) < 1:
            raise ValueError("market max dates per batch 必須 >= 1")
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_market_set",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=(
                "candidate_global_average",
                (
                    "candidate_conditioned_query_attention_pooling"
                    if candidate_conditioned
                    else "learned_query_attention_pooling"
                ),
            ),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            use_dataset_context=False,
            sequence_input_paths=("raw_level", "point_in_time_market_set"),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
            requires_market_set=True,
            market_set_history_bars=int(BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS),
            market_set_base_features=tuple(BREAKOUT_QUALITY_MARKET_SET_BASE_FEATURES),
            market_set_stock_embedding_dim=int(BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM),
            market_set_temporal_channels=int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS),
            market_set_temporal_kernel_size=int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE),
            market_set_temporal_stride=int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE),
            market_set_temporal_dilations=tuple(
                int(value) for value in BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS
            ),
            market_set_temporal_normalization=str(
                BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION
            ),
            market_set_temporal_normalization_groups=int(
                BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS
            ),
            market_set_query_mode=("candidate_conditioned" if candidate_conditioned else None),
            market_set_query_count=query_count,
            market_set_attention_heads=int(BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS),
            market_set_embedding_dim=int(BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM),
            market_set_fusion_hidden_dim=int(BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM),
            market_set_min_valid_history_ratio=float(
                BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO
            ),
            market_set_max_stocks=int(BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS),
            market_set_max_dates_per_batch=int(BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH),
        )

    if normalized == INCEPTION_TIME_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average",),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_conditional_mfe_safety",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average", "primary_mfe_head", "conditional_safety_head"),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_safety_conditional_mfe",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average", "raw_safety_head", "conditional_mfe_head"),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_safety_raw_mfe_hmhs",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average", "raw_safety_head", "safety_conditioned_raw_mfe_head", "direct_hmhs_head"),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        latent_width = 32 * (len(kernel_sizes) + 1)
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_safety_raw_mfe_hmhs_mlp",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=(
                "global_average",
                "raw_safety_head",
                "safety_conditioned_raw_mfe_head",
                "direct_hmhs_mlp_head",
            ),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            head_width=latent_width,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_RISK_CONTEXT_V1:
        depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
        kernel_sizes = build_breakout_quality_inception_kernel_sizes()
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time_risk_context",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average", "risk_context_mlp_concat"),
            dropout=0.0,
            receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
            normalization="batch_norm",
            normalization_groups=None,
            head_width=16,
            use_dataset_context=True,
            sequence_input_paths=("raw_level", "universal_risk_economic_context"),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
        )

    if normalized == INCEPTION_TIME_GROUP_NORM_V1:
        # Legacy 9A-GN 必須維持原始固定結構，確保舊 checkpoint／manifest 可重建。
        depth = 6
        kernel_sizes = (39, 19, 9)
        return BreakoutQualityModelSpec(
            architecture=normalized,
            family="inception_time",
            channels=32,
            kernel_size=max(kernel_sizes),
            dilations=(),
            convolutions_per_block=1,
            pooling=("global_average",),
            dropout=0.0,
            receptive_field_bars=1 + depth * (max(kernel_sizes) - 1),
            normalization="group_norm",
            normalization_groups=8,
            use_dataset_context=False,
            sequence_input_paths=("raw_level",),
            inception_depth=depth,
            inception_filters=32,
            inception_bottleneck_channels=32,
            inception_kernel_sizes=kernel_sizes,
            inception_residual_every=3,
        )

    kernel_size = 3
    dilations = (1, 2, 4, 8, 16, 32)
    convolutions_per_block = 2
    return BreakoutQualityModelSpec(
        architecture=RESIDUAL_TCN_V1,
        family="residual_tcn",
        channels=32,
        kernel_size=kernel_size,
        dilations=dilations,
        convolutions_per_block=convolutions_per_block,
        pooling=("last", "average", "max"),
        dropout=0.20,
        receptive_field_bars=_residual_receptive_field(
            kernel_size=kernel_size,
            dilations=dilations,
            convolutions_per_block=convolutions_per_block,
        ),
    )


def validate_model_sequence_length(
    model_spec: BreakoutQualityModelSpec, sequence_length: int
) -> None:
    normalized_length = int(sequence_length)
    if normalized_length < 1:
        raise ValueError("model sequence_length 必須 >= 1")
    if bool(model_spec.requires_market_set):
        market_history = int(model_spec.market_set_history_bars or 0)
        if market_history != normalized_length:
            raise ValueError(
                "Market Set 第一版要求 candidate sequence_length 與 market history 一致: "
                f"candidate={normalized_length}, market={market_history}"
            )
    if model_spec.family != "patch_transformer":
        return
    if normalized_length < 1:
        raise ValueError("Patch Transformer sequence_length 必須 >= 1")
    if model_spec.family == "patch_transformer":
        patch_size = int(model_spec.patch_transformer_patch_size or 0)
        patch_stride = int(model_spec.patch_transformer_patch_stride or 0)
        if patch_size < 1 or patch_stride != patch_size:
            raise ValueError("Patch Transformer 必須使用有效的非重疊 patch spec")
        if normalized_length < patch_size or normalized_length % patch_size != 0:
            raise ValueError(
                "Patch Transformer sequence_length 必須可被 patch size 整除: "
                f"sequence_length={normalized_length}, patch_size={patch_size}"
            )


def model_spec_from_manifest(payload: Mapping[str, object]) -> BreakoutQualityModelSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("breakout quality model_spec 必須是 object")
    architecture = normalize_model_architecture(str(payload.get("architecture", "")))
    expected = get_model_spec(architecture)
    actual = dict(payload)
    if actual != expected.as_manifest_payload():
        raise ValueError(
            "breakout quality model_spec 與版本化正式規格不一致: "
            f"architecture={architecture}, expected={expected.as_manifest_payload()}, actual={actual}"
        )
    return expected


__all__ = [
    "ACTIVE_MODEL_ARCHITECTURES",
    "BreakoutQualityModelSpec",
    "INCEPTION_TIME_GROUP_NORM_V1",
    "INCEPTION_TIME_MARKET_SET_CANDIDATE_V1",
    "INCEPTION_TIME_MARKET_SET_V1",
    "INCEPTION_TIME_V1",
    "INCEPTION_TIME_RISK_CONTEXT_V1",
    "INCEPTION_TIME_CONDITIONAL_MFE_SAFETY_V1",
    "INCEPTION_TIME_SAFETY_CONDITIONAL_MFE_V1",
    "INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_V1",
    "INCEPTION_TIME_SAFETY_RAW_MFE_HMHS_MLP_V1",
    "LEGACY_MODEL_ARCHITECTURES",
    "MULTISCALE_CNN_V1",
    "MULTISCALE_CNN_V2",
    "MULTISCALE_CNN_V3",
    "MULTISCALE_CNN_V4",
    "MULTISCALE_CNN_V5",
    "MULTISCALE_CNN_V6",
    "MULTISCALE_CNN_V7",
    "MULTISCALE_CNN_V8",
    "MULTISCALE_CNN_REGIME_CONTEXT_V1",
    "MULTISCALE_CNN_SEQUENCE_ONLY_V1",
    "MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1",
    "MODERN_TCN_V1",
    "MANTIS_V2_FROZEN_LINEAR_V1",
    "MOMENT_1_BASE_FROZEN_LINEAR_V1",
    "PATCH_TRANSFORMER_V1",
    "TS2VEC_FROZEN_LINEAR_V1",
    "RESIDUAL_TCN_V1",
    "SUPPORTED_MODEL_ARCHITECTURES",
    "TINY_CNN_V1",
    "get_model_spec",
    "model_spec_from_manifest",
    "normalize_active_model_architecture",
    "normalize_model_architecture",
    "validate_model_sequence_length",
]
