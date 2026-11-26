"""Architecture-family builders for breakout-quality model specifications.

Builders own architecture-specific construction.  The public resolver lives in
``spec_registry`` so adding an architecture does not extend a central switch.
"""

from __future__ import annotations

from dataclasses import replace

from collections.abc import Callable

from config.breakout_quality import (
    BREAKOUT_QUALITY_INCEPTION_DEPTH,
    BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY,
)
from core.breakout_quality_policy import (
    build_breakout_quality_inception_kernel_sizes,
    resolve_breakout_quality_inception_receptive_field_bars,
)
from filters.breakout_quality.models.architectures import *
from filters.breakout_quality.models.spec_contract import BreakoutQualityModelSpec

ModelSpecBuilder = Callable[[str], BreakoutQualityModelSpec]


def _build_inception_spec(
    architecture: str,
    *,
    family: str,
    pooling: tuple[str, ...],
    use_dataset_context: bool = False,
    sequence_input_paths: tuple[str, ...] = ("raw_level",),
    head_width: int | None = None,
) -> BreakoutQualityModelSpec:
    descriptor_options = get_architecture_descriptor(architecture).spec_options_dict()
    depth = int(descriptor_options.get("inception_depth", BREAKOUT_QUALITY_INCEPTION_DEPTH))
    filters = int(descriptor_options.get("inception_filters", 32))
    bottleneck_channels = int(descriptor_options.get("inception_bottleneck_channels", 32))
    input_window_bars_value = descriptor_options.get("input_window_bars")
    input_window_bars = None if input_window_bars_value is None else int(input_window_bars_value)
    if filters < 1 or bottleneck_channels < 1:
        raise ValueError("InceptionTime descriptor width必須為正整數")
    if input_window_bars is not None and input_window_bars < 1:
        raise ValueError("InceptionTime descriptor input_window_bars必須為正整數")
    declared_kernel_sizes = tuple(int(value) for value in descriptor_options.get("inception_kernel_sizes", ()))
    kernel_sizes = declared_kernel_sizes or build_breakout_quality_inception_kernel_sizes()
    if not kernel_sizes or any(value < 1 or value % 2 == 0 for value in kernel_sizes):
        raise ValueError("InceptionTime descriptor kernels必須是非空正奇數")
    residual_every = int(
        descriptor_options.get("inception_residual_every", BREAKOUT_QUALITY_INCEPTION_RESIDUAL_EVERY)
    )
    if depth < 1 or residual_every < 1 or depth % residual_every != 0:
        raise ValueError("InceptionTime descriptor depth/residual interval必須為可整除的正整數")
    declared_module_dilations = tuple(
        int(value)
        for value in descriptor_options.get("inception_module_dilations", ())
    )
    module_dilations = declared_module_dilations or (1,) * depth
    if len(module_dilations) != depth or any(value < 1 for value in module_dilations):
        raise ValueError("InceptionTime module dilations必須與depth同長且皆為正整數")
    return BreakoutQualityModelSpec(
        architecture=architecture,
        family=family,
        channels=filters,
        kernel_size=max(kernel_sizes),
        dilations=(),
        convolutions_per_block=1,
        pooling=pooling,
        dropout=0.0,
        receptive_field_bars=1 + (max(kernel_sizes) - 1) * sum(module_dilations),
        input_window_bars=input_window_bars,
        normalization="batch_norm",
        normalization_groups=None,
        head_width=head_width,
        use_dataset_context=use_dataset_context,
        sequence_input_paths=sequence_input_paths,
        stock_close_ema_spans=tuple(descriptor_options.get("stock_close_ema_spans", ())),
        calendar_bar_windows=tuple(descriptor_options.get("calendar_bar_windows", ())),
        calendar_encoder_channels=descriptor_options.get("calendar_encoder_channels"),
        calendar_encoder_depth=descriptor_options.get("calendar_encoder_depth"),
        calendar_encoder_kernel_size=descriptor_options.get("calendar_encoder_kernel_size"),
        calendar_fusion_width=descriptor_options.get("calendar_fusion_width"),
        inception_depth=depth,
        inception_filters=filters,
        inception_bottleneck_channels=bottleneck_channels,
        inception_kernel_sizes=kernel_sizes,
        inception_module_dilations=declared_module_dilations,
        inception_residual_every=residual_every,
    )


def build_inception_variant_spec(architecture: str) -> BreakoutQualityModelSpec:
    options = get_architecture_descriptor(architecture).spec_options_dict()
    spec = _build_inception_spec(
        architecture,
        family=str(options["family"]),
        pooling=tuple(options["pooling"]),
        use_dataset_context=bool(options["use_dataset_context"]),
        sequence_input_paths=tuple(options["sequence_input_paths"]),
        head_width=options.get("head_width"),
    )
    output_classes = options.get("classifier_output_classes")
    if output_classes is not None:
        if int(output_classes) < 2:
            raise ValueError("categorical classifier width must be at least 2")
        spec = replace(spec, classifier_output_classes=int(output_classes))
    return spec
