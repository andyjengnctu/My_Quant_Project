"""Breakout-quality model specification data contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

    def final_mfe_topology_contract(self) -> dict[str, str] | None:
        """Return canonical final-MFE topology semantics without changing model identity."""

        heads = set(self.pooling)
        if "safety_patch_token_global_average" in heads and "mfe_inception_global_average" in heads:
            return {
                "architecture": "independent_patch_transformer_safety_encoder_plus_inceptiontime_conditional_mfe_encoder",
                "conditional_mfe_head_inputs": "inceptiontime_latent_only_no_predicted_safety_context",
                "gradient_ownership": "safety_loss_updates_patch_encoder_only_conditional_mfe_loss_updates_inceptiontime_encoder_only",
            }
        if "task_specific_final_residual_group" in heads and "raw_mfe_head" in heads:
            return {
                "architecture": "shared_low_level_encoder_task_specific_safety_and_mfe_final_residual_groups",
                "mfe_head_inputs": "mfe_specific_latent_only_no_safety_prediction_input",
            }
        if "raw_mfe_head" in heads:
            return {
                "architecture": "shared_encoder_independent_raw_safety_and_raw_mfe_heads",
                "mfe_head_inputs": "shared_latent_only_no_safety_prediction_input",
            }
        if "conditional_mfe_head" in heads:
            return {
                "architecture": "shared_encoder_raw_safety_head_plus_safety_conditioned_mfe_head",
                "mfe_head_inputs": "shared_latent_plus_stop_gradient_raw_safety_probability",
            }
        return None

    def hs_conditional_mfe_topology_contract(self) -> dict[str, str] | None:
        """Return topology semantics for Safety + true-HS Conditional-MFE objectives."""

        heads = set(self.pooling)
        if "safety_patch_token_global_average" in heads and "mfe_inception_global_average" in heads:
            return {
                "architecture": "independent_patch_transformer_safety_encoder_plus_inceptiontime_conditional_mfe_encoder",
                "conditional_mfe_head_inputs": "inceptiontime_latent_only_no_predicted_safety_context",
                "gradient_ownership": "safety_loss_updates_patch_encoder_only_conditional_mfe_loss_updates_inceptiontime_encoder_only",
            }
        if "task_specific_final_residual_group" in heads and "raw_mfe_head" in heads:
            contract = {
                "architecture": "shared_low_level_encoder_task_specific_safety_and_conditional_mfe_final_residual_groups",
                "conditional_mfe_head_inputs": "mfe_specific_latent_only_no_predicted_safety_context",
                "shared_encoder_gradient": "shared_lower_residual_groups_receive_both_heads_task_specific_final_groups_receive_own_head_only",
            }
            if "safety_scalar_attention_pool" in heads:
                contract["safety_pooling"] = (
                    "single_scalar_temporal_attention_over_safety_specific_feature_map"
                )
            return contract
        if "raw_mfe_head" in heads:
            return {
                "architecture": "shared_encoder_independent_raw_safety_and_conditional_mfe_heads",
                "conditional_mfe_head_inputs": "shared_latent_only_no_predicted_safety_context",
            }
        return None

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
