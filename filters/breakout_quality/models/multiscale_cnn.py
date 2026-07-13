"""Medium multi-scale 1D CNN for short-, medium-, and long-horizon breakout patterns."""

from __future__ import annotations

import math


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
    if not (
        len(branch_factors)
        == len(branch_kernels)
        == len(branch_summary_windows)
        == 3
    ):
        raise ValueError("multiscale_cnn_v1 必須固定包含三個時間尺度 branch")

    channels = int(spec.channels)
    group_count = int(spec.normalization_groups)
    if channels < 1 or group_count < 1 or channels % group_count != 0:
        raise ValueError("multiscale_cnn_v1 channels 必須可被 normalization_groups 整除")

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
        def __init__(self, *, downsample_factor: int, kernel_sizes: tuple[int, int]):
            super().__init__()
            if int(downsample_factor) < 1:
                raise ValueError("multiscale downsample factor 必須 >= 1")
            if len(kernel_sizes) != 2 or any(int(value) < 1 or int(value) % 2 == 0 for value in kernel_sizes):
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
                CausalConv1d(int(feature_count), channels, int(kernel_sizes[0])),
                nn.GroupNorm(group_count, channels),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
                CausalConv1d(channels, channels, int(kernel_sizes[1])),
                nn.GroupNorm(group_count, channels),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
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
                    )
                    for factor, kernels in zip(branch_factors, branch_kernels)
                ]
            )
            summary_width = channels * sum(len(windows) for windows in branch_summary_windows)
            self.head = nn.Sequential(
                nn.Linear(summary_width + int(context_count), int(spec.head_width)),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
                nn.Linear(int(spec.head_width), 2),
            )

        @staticmethod
        def _summarize_branch(z, *, downsample_factor: int, windows_bars: tuple[int, ...]):
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
            sequence = x.transpose(1, 2)
            summaries = []
            for branch, factor, windows in zip(
                self.branches,
                branch_factors,
                branch_summary_windows,
            ):
                branch_output = branch(sequence)
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


__all__ = ["build_multiscale_cnn"]
