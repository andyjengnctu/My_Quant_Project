"""Architecture-family builders for breakout-quality model specifications.

Builders own architecture-specific construction.  The public resolver lives in
``spec_registry`` so adding an architecture does not extend a central switch.
"""

from __future__ import annotations

from collections.abc import Callable

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
from filters.breakout_quality.models.architectures import *
from filters.breakout_quality.models.regime_context import (
    REGIME_CONTEXT_ANNUALIZATION_BARS,
    REGIME_CONTEXT_FEATURES,
    REGIME_CONTEXT_LOOKBACK_BARS,
)
from filters.breakout_quality.models.spec_contract import BreakoutQualityModelSpec

ModelSpecBuilder = Callable[[str], BreakoutQualityModelSpec]


def _residual_receptive_field(*, kernel_size: int, dilations: tuple[int, ...], convolutions_per_block: int) -> int:
    return 1 + int(convolutions_per_block) * (int(kernel_size) - 1) * sum(int(value) for value in dilations)


def _multiscale_receptive_field(*, downsample_factors: tuple[int, ...], kernel_sizes: tuple[tuple[int, ...], ...]) -> int:
    if len(downsample_factors) != len(kernel_sizes):
        raise ValueError("multiscale branch spec 長度不一致")
    branch_fields = []
    for factor, kernels in zip(downsample_factors, kernel_sizes):
        if int(factor) < 1 or not kernels:
            raise ValueError("multiscale branch factor 與 kernels 必須有效")
        branch_fields.append(int(factor) * (1 + sum(int(kernel) - 1 for kernel in kernels)))
    return max(branch_fields)


def build_tiny_cnn_spec(architecture: str) -> BreakoutQualityModelSpec:
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family="tiny_cnn",
        channels=32,
        kernel_size=5,
        dilations=(1, 2),
        convolutions_per_block=1,
        pooling=("average",),
        dropout=0.10,
        receptive_field_bars=11,
    )


def build_multiscale_cnn_spec(architecture: str) -> BreakoutQualityModelSpec:
    downsample_factors = (1, 2, 4)
    branch_kernel_sizes = ((3, 5), (9, 15), (31, 31))
    branch_summary_windows_bars = ((0, 20), (0, 60), (120, 300))
    overrides = {
        MULTISCALE_CNN_V1: {},
        MULTISCALE_CNN_V2: {"branch_input_representations": ("return_delta", "return_delta", "level")},
        MULTISCALE_CNN_V3: {"branch_input_representations": ("market_relative_return_delta", "market_relative_return_delta", "level")},
        MULTISCALE_CNN_V4: {"branch_channels": (16, 16, 8)},
        MULTISCALE_CNN_V5: {"branch_channels": (16, 16, 12)},
        MULTISCALE_CNN_V6: {"branch_dropouts": (0.25, 0.25, 0.40)},
        MULTISCALE_CNN_V7: {"branch_input_representations": ("return_delta", "level", "level")},
        MULTISCALE_CNN_V8: {"branch_input_representations": ("level", "return_delta", "level")},
        MULTISCALE_CNN_REGIME_CONTEXT_V1: {
            "derived_context_features": REGIME_CONTEXT_FEATURES,
            "derived_context_lookback_bars": REGIME_CONTEXT_LOOKBACK_BARS,
            "derived_context_annualization_bars": REGIME_CONTEXT_ANNUALIZATION_BARS,
        },
        MULTISCALE_CNN_SEQUENCE_ONLY_V1: {"use_dataset_context": False},
        MULTISCALE_CNN_SEQUENCE_ONLY_DUAL_PATH_V1: {
            "use_dataset_context": False,
            "sequence_input_paths": ("raw_level", "window_zscore"),
            "window_normalization_epsilon": 1e-5,
        },
    }
    if architecture not in overrides:
        raise ValueError(f"未註冊的 multiscale architecture: {architecture}")
    return BreakoutQualityModelSpec(
        architecture=architecture,
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
        **overrides[architecture],
    )


def _build_patch_transformer_spec(architecture: str, *, family: str, pooling: tuple[str, ...], head_width: int | None = None) -> BreakoutQualityModelSpec:
    patch_size = 10
    embedding_dim = 128
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family=family,
        channels=embedding_dim,
        kernel_size=patch_size,
        dilations=(),
        convolutions_per_block=1,
        pooling=pooling,
        dropout=0.10,
        receptive_field_bars=300,
        normalization="layer_norm",
        head_width=head_width,
        use_dataset_context=False,
        sequence_input_paths=("raw_level_temporal_nonoverlap_patches",),
        patch_transformer_patch_size=patch_size,
        patch_transformer_patch_stride=patch_size,
        patch_transformer_embedding_dim=embedding_dim,
        patch_transformer_depth=3,
        patch_transformer_heads=4,
        patch_transformer_mlp_dim=256,
        patch_transformer_pooling="mean",
        patch_transformer_positional_encoding="sinusoidal",
    )


def build_patch_transformer_spec(architecture: str) -> BreakoutQualityModelSpec:
    return _build_patch_transformer_spec(
        architecture,
        family="patch_transformer",
        pooling=("patch_mean",),
    )


def build_patch_token_ranker_spec(architecture: str) -> BreakoutQualityModelSpec:
    return _build_patch_transformer_spec(
        architecture,
        family="patch_token_transformer_ranker",
        pooling=("patch_token_global_average", "single_rank_head"),
    )


def build_patch_token_joint_spec(architecture: str) -> BreakoutQualityModelSpec:
    return _build_patch_transformer_spec(
        architecture,
        family="patch_token_transformer_safety_raw_mfe_joint_attn_mlp",
        pooling=(
            "patch_token_global_average_for_marginal_heads",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "joint_scalar_attention_pool_over_patch_tokens",
            "joint_mlp_head",
        ),
        head_width=128,
    )


def build_moment_spec(architecture: str) -> BreakoutQualityModelSpec:
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
        architecture=architecture,
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


def build_mantis_spec(architecture: str) -> BreakoutQualityModelSpec:
    from filters.breakout_quality.mantis_contract import MANTIS_V2_REPOSITORY, MANTIS_V2_REVISION
    return BreakoutQualityModelSpec(
        architecture=architecture,
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


def build_ts2vec_spec(architecture: str) -> BreakoutQualityModelSpec:
    depth, hidden_dims, output_dims, kernel_size = 8, 128, 320, 3
    return BreakoutQualityModelSpec(
        architecture=architecture,
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


def _build_modern_tcn_spec(architecture: str, *, joint: bool) -> BreakoutQualityModelSpec:
    depth, channels, kernel_size, expansion_ratio = 6, 96, 51, 4
    if joint:
        family = "modern_tcn_safety_raw_mfe_joint_attn_mlp"
        pooling = (
            "global_average_for_marginal_heads",
            "raw_safety_head",
            "safety_conditioned_raw_mfe_head",
            "joint_scalar_attention_pool",
            "joint_mlp_head",
        )
        head_width = channels
    else:
        family = "modern_tcn"
        pooling = ("global_average",)
        head_width = None
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family=family,
        channels=channels,
        kernel_size=kernel_size,
        dilations=(),
        convolutions_per_block=1,
        pooling=pooling,
        dropout=0.10,
        receptive_field_bars=1 + depth * (kernel_size - 1),
        normalization="batch_norm",
        head_width=head_width,
        use_dataset_context=False,
        sequence_input_paths=("raw_level",),
        modern_tcn_depth=depth,
        modern_tcn_channels=channels,
        modern_tcn_kernel_size=kernel_size,
        modern_tcn_expansion_ratio=expansion_ratio,
    )


def build_modern_tcn_spec(architecture: str) -> BreakoutQualityModelSpec:
    return _build_modern_tcn_spec(architecture, joint=False)


def build_modern_tcn_joint_spec(architecture: str) -> BreakoutQualityModelSpec:
    return _build_modern_tcn_spec(architecture, joint=True)


def _build_inception_spec(
    architecture: str,
    *,
    family: str,
    pooling: tuple[str, ...],
    use_dataset_context: bool = False,
    sequence_input_paths: tuple[str, ...] = ("raw_level",),
    head_width: int | None = None,
) -> BreakoutQualityModelSpec:
    depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
    kernel_sizes = build_breakout_quality_inception_kernel_sizes()
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family=family,
        channels=32,
        kernel_size=max(kernel_sizes),
        dilations=(),
        convolutions_per_block=1,
        pooling=pooling,
        dropout=0.0,
        receptive_field_bars=resolve_breakout_quality_inception_receptive_field_bars(),
        normalization="batch_norm",
        normalization_groups=None,
        head_width=head_width,
        use_dataset_context=use_dataset_context,
        sequence_input_paths=sequence_input_paths,
        inception_depth=depth,
        inception_filters=32,
        inception_bottleneck_channels=32,
        inception_kernel_sizes=kernel_sizes,
        inception_residual_every=int(BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY),
    )



def build_inception_variant_spec(architecture: str) -> BreakoutQualityModelSpec:
    family, pooling, use_context, paths, head_width = INCEPTION_VARIANT_SPECS[architecture]
    return _build_inception_spec(
        architecture,
        family=family,
        pooling=pooling,
        use_dataset_context=use_context,
        sequence_input_paths=paths,
        head_width=head_width,
    )


def build_inception_hmhs_mlp_spec(architecture: str) -> BreakoutQualityModelSpec:
    kernel_sizes = build_breakout_quality_inception_kernel_sizes()
    return _build_inception_spec(
        architecture,
        family="inception_time_safety_raw_mfe_hmhs_mlp",
        pooling=("global_average", "raw_safety_head", "safety_conditioned_raw_mfe_head", "direct_hmhs_mlp_head"),
        head_width=32 * (len(kernel_sizes) + 1),
    )


def build_inception_joint_spec(architecture: str) -> BreakoutQualityModelSpec:
    kernel_sizes = build_breakout_quality_inception_kernel_sizes()
    return _build_inception_spec(
        architecture,
        family="inception_time_safety_raw_mfe_joint_attn_mlp",
        pooling=("global_average_for_marginal_heads", "raw_safety_head", "safety_conditioned_raw_mfe_head", "joint_scalar_attention_pool", "joint_mlp_head"),
        head_width=32 * (len(kernel_sizes) + 1),
    )


def build_inception_group_norm_spec(architecture: str) -> BreakoutQualityModelSpec:
    depth, kernel_sizes = 6, (39, 19, 9)
    return BreakoutQualityModelSpec(
        architecture=architecture,
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


def build_market_set_spec(architecture: str) -> BreakoutQualityModelSpec:
    depth = int(BREAKOUT_QUALITY_INCEPTION_DEPTH)
    kernel_sizes = build_breakout_quality_inception_kernel_sizes()
    candidate_conditioned = architecture == INCEPTION_TIME_MARKET_SET_CANDIDATE_V1
    query_count = int(BREAKOUT_QUALITY_MARKET_SET_CANDIDATE_QUERY_COUNT if candidate_conditioned else BREAKOUT_QUALITY_MARKET_SET_QUERY_COUNT)
    if int(BREAKOUT_QUALITY_MARKET_SET_HISTORY_BARS) < 2:
        raise ValueError("MARKET_SET_HISTORY_BARS 必須 >= 2")
    if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_KERNEL_SIZE) < 1:
        raise ValueError("MARKET_SET_TEMPORAL_KERNEL_SIZE 必須 >= 1")
    if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_STRIDE) < 1:
        raise ValueError("MARKET_SET_TEMPORAL_STRIDE 必須 >= 1")
    if not BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS or any(int(value) < 1 for value in BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS):
        raise ValueError("MARKET_SET_TEMPORAL_DILATIONS 必須是非空正整數")
    if query_count < 1:
        raise ValueError("market query count 必須 >= 1")
    if int(BREAKOUT_QUALITY_MARKET_SET_STOCK_EMBEDDING_DIM) % int(BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS) != 0:
        raise ValueError("market stock embedding dim 必須可被 attention heads 整除")
    if str(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION) != "group_norm":
        raise ValueError("market temporal normalization 第一版固定為 group_norm")
    if int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_CHANNELS) % int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS) != 0:
        raise ValueError("market temporal channels 必須可被 normalization groups 整除")
    if not 0.0 < float(BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO) <= 1.0:
        raise ValueError("market min valid history ratio 必須介於 0 與 1")
    if int(BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS) < 0:
        raise ValueError("market max stocks 必須 >= 0")
    if int(BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH) < 1:
        raise ValueError("market max dates per batch 必須 >= 1")
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family="inception_time_market_set",
        channels=32,
        kernel_size=max(kernel_sizes),
        dilations=(),
        convolutions_per_block=1,
        pooling=("candidate_global_average", "candidate_conditioned_query_attention_pooling" if candidate_conditioned else "learned_query_attention_pooling"),
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
        market_set_temporal_dilations=tuple(int(value) for value in BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_DILATIONS),
        market_set_temporal_normalization=str(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION),
        market_set_temporal_normalization_groups=int(BREAKOUT_QUALITY_MARKET_SET_TEMPORAL_NORMALIZATION_GROUPS),
        market_set_query_mode="candidate_conditioned" if candidate_conditioned else None,
        market_set_query_count=query_count,
        market_set_attention_heads=int(BREAKOUT_QUALITY_MARKET_SET_ATTENTION_HEADS),
        market_set_embedding_dim=int(BREAKOUT_QUALITY_MARKET_SET_EMBEDDING_DIM),
        market_set_fusion_hidden_dim=int(BREAKOUT_QUALITY_MARKET_SET_FUSION_HIDDEN_DIM),
        market_set_min_valid_history_ratio=float(BREAKOUT_QUALITY_MARKET_SET_MIN_VALID_HISTORY_RATIO),
        market_set_max_stocks=int(BREAKOUT_QUALITY_MARKET_SET_MAX_STOCKS),
        market_set_max_dates_per_batch=int(BREAKOUT_QUALITY_MARKET_SET_MAX_DATES_PER_BATCH),
    )


def build_residual_tcn_spec(architecture: str) -> BreakoutQualityModelSpec:
    kernel_size, dilations, convolutions_per_block = 3, (1, 2, 4, 8, 16, 32), 2
    return BreakoutQualityModelSpec(
        architecture=architecture,
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
