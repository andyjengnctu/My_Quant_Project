"""Versioned model specifications for breakout quality classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
    RESIDUAL_TCN_V1,
)
ACTIVE_MODEL_ARCHITECTURES = (
    MULTISCALE_CNN_V1,
    MULTISCALE_CNN_REGIME_CONTEXT_V1,
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
    derived_context_features: tuple[str, ...] = ()
    derived_context_lookback_bars: tuple[int, ...] = ()
    derived_context_annualization_bars: int | None = None

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
    }:
        downsample_factors = (1, 2, 4)
        branch_kernel_sizes = ((3, 5), (9, 15), (31, 31))
        branch_summary_windows_bars = ((0, 20), (0, 60), (120, 300))
        branch_channels = ()
        branch_dropouts = ()
        derived_context_features = ()
        derived_context_lookback_bars = ()
        derived_context_annualization_bars = None
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
            derived_context_features=derived_context_features,
            derived_context_lookback_bars=derived_context_lookback_bars,
            derived_context_annualization_bars=derived_context_annualization_bars,
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
    "RESIDUAL_TCN_V1",
    "SUPPORTED_MODEL_ARCHITECTURES",
    "TINY_CNN_V1",
    "get_model_spec",
    "model_spec_from_manifest",
    "normalize_active_model_architecture",
    "normalize_model_architecture",
]
