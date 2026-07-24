"""Medium multi-scale 1D CNN for short-, medium-, and long-horizon breakout patterns."""

from __future__ import annotations

import math

from filters.breakout_quality.models.regime_context import (
    REGIME_CONTEXT_FEATURES,
    build_regime_context_from_level_sequence,
)


EXPECTED_OHLCV_FEATURE_COUNT = 10
RETURN_DELTA_REPRESENTATION = "return_delta"
MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION = "market_relative_return_delta"
LEVEL_REPRESENTATION = "level"
RAW_LEVEL_INPUT_PATH = "raw_level"
WINDOW_ZSCORE_INPUT_PATH = "window_zscore"
SUPPORTED_SEQUENCE_INPUT_PATHS = frozenset(
    {RAW_LEVEL_INPUT_PATH, WINDOW_ZSCORE_INPUT_PATH}
)
SUPPORTED_BRANCH_INPUT_REPRESENTATIONS = frozenset(
    {
        LEVEL_REPRESENTATION,
        RETURN_DELTA_REPRESENTATION,
        MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION,
    }
)


def build_return_delta_representation(torch, sequence):
    """Convert canonical normalized OHLCV levels into stationary one-bar changes.

    Input and output both use [batch, feature, time] and preserve the canonical
    10-column order. Price outputs are log(open/high/low/close) relative to the
    previous close; volume outputs are first differences of the existing robust
    log-volume normalization. The first bar is zero because its prior bar is
    outside the stored feature window.
    """

    if int(sequence.ndim) != 3:
        raise ValueError("return/delta representation 需要 [batch, feature, time] tensor")
    if int(sequence.shape[1]) != EXPECTED_OHLCV_FEATURE_COUNT:
        raise ValueError(
            "multiscale Return／Delta 表示需要 canonical 10-column OHLCV feature contract"
        )

    result = torch.zeros_like(sequence)
    if int(sequence.shape[2]) <= 1:
        return result

    epsilon = 1e-6

    def _log_price(feature_index: int):
        normalized = torch.clamp(sequence[:, int(feature_index), :], min=-1.0 + epsilon)
        return torch.log1p(normalized)

    for base_index in (0, 5):
        previous_close = _log_price(base_index + 3)[:, :-1]
        for offset in (0, 1, 2, 3):
            current_price = _log_price(base_index + offset)[:, 1:]
            result[:, base_index + offset, 1:] = current_price - previous_close
        volume_index = base_index + 4
        result[:, volume_index, 1:] = (
            sequence[:, volume_index, 1:] - sequence[:, volume_index, :-1]
        )

    return result


def build_market_relative_return_delta_representation(torch, sequence):
    """Replace stock price changes with stock-minus-0050 relative changes.

    The canonical 10-column output contract is preserved:
    stock O/H/L/C price channels become their one-bar changes minus the matching
    0050 changes; stock volume delta, all 0050 price changes, and 0050 volume
    delta remain unchanged. Input and output both use [batch, feature, time].
    """

    result = build_return_delta_representation(torch, sequence)
    relative = result.clone()
    relative[:, 0:4, :] = result[:, 0:4, :] - result[:, 5:9, :]
    return relative


def build_window_zscore_representation(torch, sequence, *, epsilon: float):
    """Normalize each sample/channel with statistics from its known input window.

    Input and output use [batch, feature, time]. The full 300-bar event window is
    available when the breakout decision is made, so the transform introduces no
    bars after the event date. Constant channels become exact zeros.
    """

    if int(sequence.ndim) != 3:
        raise ValueError("window normalization 需要 [batch, feature, time] tensor")
    normalized_epsilon = float(epsilon)
    if not math.isfinite(normalized_epsilon) or normalized_epsilon <= 0.0:
        raise ValueError("window normalization epsilon 必須是有限正數")
    mean = torch.mean(sequence, dim=2, keepdim=True)
    centered = sequence - mean
    variance = torch.mean(centered * centered, dim=2, keepdim=True)
    scale = torch.sqrt(torch.clamp(variance, min=normalized_epsilon**2))
    return centered / scale


def build_multiscale_cnn(nn, torch, *, feature_count: int, context_count: int, spec):
    branch_factors = tuple(int(value) for value in spec.branch_downsample_factors)
    branch_kernels = tuple(
        tuple(int(kernel) for kernel in kernels)
        for kernels in spec.branch_kernel_sizes
    )
    branch_summary_windows = tuple(
        tuple(int(window) for window in windows)
        for windows in spec.branch_summary_windows_bars
    )
    branch_input_representations = tuple(
        str(value).strip().lower()
        for value in (
            spec.branch_input_representations
            or (LEVEL_REPRESENTATION,) * len(branch_factors)
        )
    )
    sequence_input_paths = tuple(
        str(value).strip().lower()
        for value in (spec.sequence_input_paths or (RAW_LEVEL_INPUT_PATH,))
    )
    invalid_input_paths = sorted(
        set(sequence_input_paths) - SUPPORTED_SEQUENCE_INPUT_PATHS
    )
    if invalid_input_paths:
        raise ValueError(
            "multiscale sequence input path 不支援: " + ", ".join(invalid_input_paths)
        )
    if not sequence_input_paths or sequence_input_paths[0] != RAW_LEVEL_INPUT_PATH:
        raise ValueError("multiscale sequence input paths 第一條必須是 raw_level")
    if len(sequence_input_paths) != len(set(sequence_input_paths)):
        raise ValueError("multiscale sequence input paths 不可重複")
    use_window_zscore_path = WINDOW_ZSCORE_INPUT_PATH in sequence_input_paths
    window_normalization_epsilon = spec.window_normalization_epsilon
    if use_window_zscore_path:
        if len(sequence_input_paths) != 2:
            raise ValueError("dual-path architecture 必須固定使用 raw_level + window_zscore")
        if window_normalization_epsilon is None:
            raise ValueError("window_zscore path 缺少 normalization epsilon")
        normalized_epsilon = float(window_normalization_epsilon)
        if not math.isfinite(normalized_epsilon) or normalized_epsilon <= 0.0:
            raise ValueError("window normalization epsilon 必須是有限正數")
    elif window_normalization_epsilon is not None:
        raise ValueError("未使用 window_zscore path 時不可設定 normalization epsilon")
    if use_window_zscore_path and any(
        representation != LEVEL_REPRESENTATION
        for representation in branch_input_representations
    ):
        raise ValueError("window_zscore dual path 目前只允許三個 Level branches")
    if not (
        len(branch_factors)
        == len(branch_kernels)
        == len(branch_summary_windows)
        == len(branch_input_representations)
        == 3
    ):
        raise ValueError("multiscale CNN 必須固定包含三個時間尺度 branch")
    invalid_representations = sorted(
        set(branch_input_representations) - SUPPORTED_BRANCH_INPUT_REPRESENTATIONS
    )
    if invalid_representations:
        raise ValueError(
            "multiscale branch input representation 不支援: "
            + ", ".join(invalid_representations)
        )
    if (
        any(
            representation != LEVEL_REPRESENTATION
            for representation in branch_input_representations
        )
        and int(feature_count) != EXPECTED_OHLCV_FEATURE_COUNT
    ):
        raise ValueError(
            "multiscale Return／Delta 表示需要 canonical 10-column OHLCV feature contract"
        )

    channels = int(spec.channels)
    branch_channels = tuple(
        int(value)
        for value in (spec.branch_channels or (channels,) * len(branch_factors))
    )
    branch_dropouts = tuple(
        float(value)
        for value in (
            spec.branch_dropouts
            or (float(spec.dropout),) * len(branch_factors)
        )
    )
    use_dataset_context = bool(spec.use_dataset_context)
    derived_context_features = tuple(str(value) for value in spec.derived_context_features)
    derived_context_lookbacks = tuple(
        int(value) for value in spec.derived_context_lookback_bars
    )
    derived_context_annualization = spec.derived_context_annualization_bars
    if derived_context_features:
        if derived_context_features != REGIME_CONTEXT_FEATURES:
            raise ValueError("multiscale derived context 欄位與正式 regime contract 不一致")
        if int(feature_count) != EXPECTED_OHLCV_FEATURE_COUNT:
            raise ValueError(
                "multiscale regime context 需要 canonical 10-column OHLCV feature contract"
            )
        if not derived_context_lookbacks or derived_context_annualization is None:
            raise ValueError("multiscale regime context spec 缺少 lookback／annualization")
    group_count = int(spec.normalization_groups)
    if len(branch_channels) != len(branch_factors):
        raise ValueError("multiscale branch_channels 長度必須等於 branch 數")
    if len(branch_dropouts) != len(branch_factors):
        raise ValueError("multiscale branch_dropouts 長度必須等於 branch 數")
    if any(value < 0.0 or value >= 1.0 for value in branch_dropouts):
        raise ValueError("multiscale branch dropout 必須位於 [0, 1)")
    if group_count < 1 or any(
        value < 1 or value % group_count != 0 for value in branch_channels
    ):
        raise ValueError(
            "multiscale CNN 每個 branch channel 數必須可被 normalization_groups 整除"
        )

    class CausalConv1d(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
            super().__init__()
            self.left_padding = int(kernel_size) - 1
            self.conv = nn.Conv1d(
                int(in_channels),
                int(out_channels),
                kernel_size=int(kernel_size),
                padding=0,
            )

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (self.left_padding, 0)))

    class MultiScaleBranch(nn.Module):
        def __init__(
            self,
            *,
            downsample_factor: int,
            kernel_sizes: tuple[int, int],
            output_channels: int,
            dropout: float,
        ):
            super().__init__()
            if int(downsample_factor) < 1:
                raise ValueError("multiscale downsample factor 必須 >= 1")
            if len(kernel_sizes) != 2 or any(
                int(value) < 1 or int(value) % 2 == 0 for value in kernel_sizes
            ):
                raise ValueError("multiscale branch 必須使用兩個正奇數 kernel")
            self.downsample_factor = int(downsample_factor)
            self.downsample = (
                nn.Identity()
                if self.downsample_factor == 1
                else nn.AvgPool1d(
                    kernel_size=self.downsample_factor,
                    stride=self.downsample_factor,
                    ceil_mode=True,
                )
            )
            self.network = nn.Sequential(
                CausalConv1d(
                    int(feature_count), int(output_channels), int(kernel_sizes[0])
                ),
                nn.GroupNorm(group_count, int(output_channels)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                CausalConv1d(
                    int(output_channels), int(output_channels), int(kernel_sizes[1])
                ),
                nn.GroupNorm(group_count, int(output_channels)),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )

        def forward(self, x):
            return self.network(self.downsample(x))

    class MultiScaleBreakoutQualityCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.branches = nn.ModuleList(
                [
                    MultiScaleBranch(
                        downsample_factor=factor,
                        kernel_sizes=kernels,
                        output_channels=output_channels,
                        dropout=dropout,
                    )
                    for factor, kernels, output_channels, dropout in zip(
                        branch_factors, branch_kernels, branch_channels, branch_dropouts
                    )
                ]
            )
            summary_width = sum(
                int(output_channels) * len(windows)
                for output_channels, windows in zip(
                    branch_channels, branch_summary_windows
                )
            )
            self.head = nn.Sequential(
                nn.Linear(
                    summary_width + (int(context_count) if use_dataset_context else 0),
                    int(spec.head_width),
                ),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
                nn.Linear(int(spec.head_width), 2),
            )
            self.normalized_branches = nn.ModuleList()
            self.path_fusions = nn.ModuleList()
            if use_window_zscore_path:
                self.normalized_branches = nn.ModuleList(
                    [
                        MultiScaleBranch(
                            downsample_factor=factor,
                            kernel_sizes=kernels,
                            output_channels=output_channels,
                            dropout=dropout,
                        )
                        for factor, kernels, output_channels, dropout in zip(
                            branch_factors,
                            branch_kernels,
                            branch_channels,
                            branch_dropouts,
                        )
                    ]
                )
                for output_channels, windows in zip(
                    branch_channels, branch_summary_windows
                ):
                    branch_summary_width = int(output_channels) * len(windows)
                    fusion = nn.Linear(
                        branch_summary_width * 2, branch_summary_width, bias=True
                    )
                    nn.init.zeros_(fusion.weight)
                    nn.init.zeros_(fusion.bias)
                    with torch.no_grad():
                        fusion.weight[:, :branch_summary_width].copy_(
                            torch.eye(branch_summary_width, dtype=fusion.weight.dtype)
                        )
                    self.path_fusions.append(fusion)
            self.derived_context_projection = None
            if derived_context_features:
                self.derived_context_projection = nn.Linear(
                    len(derived_context_features),
                    int(spec.head_width),
                    bias=False,
                )
                nn.init.zeros_(self.derived_context_projection.weight)

        @staticmethod
        def _summarize_branch(
            z, *, downsample_factor: int, windows_bars: tuple[int, ...]
        ):
            summaries = []
            for window_bars in windows_bars:
                if int(window_bars) == 0:
                    summaries.append(z[:, :, -1])
                    continue
                reduced_window = max(
                    1,
                    int(math.ceil(int(window_bars) / int(downsample_factor))),
                )
                reduced_window = min(reduced_window, int(z.shape[2]))
                summaries.append(torch.mean(z[:, :, -reduced_window:], dim=2))
            return summaries

        def forward(self, x, context):
            level_sequence = x.transpose(1, 2)
            prepared_inputs = {LEVEL_REPRESENTATION: level_sequence}
            if RETURN_DELTA_REPRESENTATION in branch_input_representations:
                prepared_inputs[RETURN_DELTA_REPRESENTATION] = (
                    build_return_delta_representation(torch, level_sequence)
                )
            if (
                MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION
                in branch_input_representations
            ):
                prepared_inputs[MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION] = (
                    build_market_relative_return_delta_representation(
                        torch, level_sequence
                    )
                )

            normalized_inputs = None
            if use_window_zscore_path:
                normalized_level_sequence = build_window_zscore_representation(
                    torch,
                    level_sequence,
                    epsilon=float(window_normalization_epsilon),
                )
                normalized_inputs = {LEVEL_REPRESENTATION: normalized_level_sequence}

            summaries = []
            for branch_index, (branch, factor, windows, representation) in enumerate(
                zip(
                    self.branches,
                    branch_factors,
                    branch_summary_windows,
                    branch_input_representations,
                )
            ):
                branch_input = prepared_inputs.get(representation)
                if branch_input is None:
                    raise AssertionError(
                        f"multiscale branch representation 尚未建立: {representation}"
                    )
                branch_output = branch(branch_input)
                raw_summaries = self._summarize_branch(
                    branch_output,
                    downsample_factor=factor,
                    windows_bars=windows,
                )
                if normalized_inputs is None:
                    summaries.extend(raw_summaries)
                    continue

                normalized_branch_input = normalized_inputs.get(representation)
                if normalized_branch_input is None:
                    raise AssertionError(
                        "multiscale normalized branch representation 尚未建立: "
                        f"{representation}"
                    )
                normalized_output = self.normalized_branches[branch_index](
                    normalized_branch_input
                )
                normalized_summaries = self._summarize_branch(
                    normalized_output,
                    downsample_factor=factor,
                    windows_bars=windows,
                )
                raw_summary = torch.cat(raw_summaries, dim=1)
                normalized_summary = torch.cat(normalized_summaries, dim=1)
                summaries.append(
                    self.path_fusions[branch_index](
                        torch.cat([raw_summary, normalized_summary], dim=1)
                    )
                )
            summary = torch.cat(summaries, dim=1)
            base_head_input = (
                torch.cat([summary, context], dim=1)
                if use_dataset_context
                else summary
            )
            if self.derived_context_projection is None:
                return self.head(base_head_input)

            derived_context = build_regime_context_from_level_sequence(
                torch,
                level_sequence,
                lookback_bars=derived_context_lookbacks,
                annualization_bars=int(derived_context_annualization),
            )
            hidden = self.head[0](base_head_input)
            hidden = hidden + self.derived_context_projection(derived_context)
            for layer in self.head[1:]:
                hidden = layer(hidden)
            return hidden

    return MultiScaleBreakoutQualityCNN()


__all__ = [
    "EXPECTED_OHLCV_FEATURE_COUNT",
    "LEVEL_REPRESENTATION",
    "RAW_LEVEL_INPUT_PATH",
    "SUPPORTED_SEQUENCE_INPUT_PATHS",
    "WINDOW_ZSCORE_INPUT_PATH",
    "MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION",
    "RETURN_DELTA_REPRESENTATION",
    "SUPPORTED_BRANCH_INPUT_REPRESENTATIONS",
    "build_market_relative_return_delta_representation",
    "build_multiscale_cnn",
    "build_return_delta_representation",
    "build_window_zscore_representation",
]
