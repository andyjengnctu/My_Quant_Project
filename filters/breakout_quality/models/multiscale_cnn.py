"""Medium multi-scale 1D CNN for short-, medium-, and long-horizon breakout patterns."""

from __future__ import annotations

import math


EXPECTED_OHLCV_FEATURE_COUNT = 10
RETURN_DELTA_REPRESENTATION = "return_delta"
MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION = "market_relative_return_delta"
LEVEL_REPRESENTATION = "level"
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
                nn.Linear(summary_width + int(context_count), int(spec.head_width)),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
                nn.Linear(int(spec.head_width), 2),
            )

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

            summaries = []
            for branch, factor, windows, representation in zip(
                self.branches,
                branch_factors,
                branch_summary_windows,
                branch_input_representations,
            ):
                branch_input = prepared_inputs.get(representation)
                if branch_input is None:
                    raise AssertionError(
                        f"multiscale branch representation 尚未建立: {representation}"
                    )
                branch_output = branch(branch_input)
                summaries.extend(
                    self._summarize_branch(
                        branch_output,
                        downsample_factor=factor,
                        windows_bars=windows,
                    )
                )
            summary = torch.cat(summaries, dim=1)
            return self.head(torch.cat([summary, context], dim=1))

    return MultiScaleBreakoutQualityCNN()


__all__ = [
    "EXPECTED_OHLCV_FEATURE_COUNT",
    "LEVEL_REPRESENTATION",
    "MARKET_RELATIVE_RETURN_DELTA_REPRESENTATION",
    "RETURN_DELTA_REPRESENTATION",
    "SUPPORTED_BRANCH_INPUT_REPRESENTATIONS",
    "build_market_relative_return_delta_representation",
    "build_multiscale_cnn",
    "build_return_delta_representation",
]
