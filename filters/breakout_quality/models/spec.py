"""Versioned model specifications for breakout quality classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


TINY_CNN_V1 = "tiny_cnn_v1"
RESIDUAL_TCN_V1 = "residual_tcn_v1"
SUPPORTED_MODEL_ARCHITECTURES = (TINY_CNN_V1, RESIDUAL_TCN_V1)


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

    def as_manifest_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dilations"] = list(self.dilations)
        payload["pooling"] = list(self.pooling)
        return payload


def normalize_model_architecture(value: str) -> str:
    architecture = str(value).strip().lower()
    if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        allowed = ", ".join(SUPPORTED_MODEL_ARCHITECTURES)
        raise ValueError(
            f"不支援的 breakout quality model architecture: {value!r}；可用值: {allowed}"
        )
    return architecture


def _residual_receptive_field(
    *, kernel_size: int, dilations: tuple[int, ...], convolutions_per_block: int
) -> int:
    return 1 + int(convolutions_per_block) * (int(kernel_size) - 1) * sum(
        int(value) for value in dilations
    )


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
    "BreakoutQualityModelSpec",
    "RESIDUAL_TCN_V1",
    "SUPPORTED_MODEL_ARCHITECTURES",
    "TINY_CNN_V1",
    "get_model_spec",
    "model_spec_from_manifest",
    "normalize_model_architecture",
]
