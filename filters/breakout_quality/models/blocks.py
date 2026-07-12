"""Reusable temporal convolution blocks for breakout quality models."""

from __future__ import annotations


def build_residual_temporal_block(nn, torch, *, channels: int, kernel_size: int, dilation: int, dropout: float):
    class CausalConv1d(nn.Module):
        def __init__(self):
            super().__init__()
            self.left_padding = int(dilation) * (int(kernel_size) - 1)
            self.conv = nn.Conv1d(
                int(channels),
                int(channels),
                kernel_size=int(kernel_size),
                dilation=int(dilation),
                padding=0,
            )

        def forward(self, x):
            return self.conv(torch.nn.functional.pad(x, (self.left_padding, 0)))

    class ResidualTemporalBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = CausalConv1d()
            self.norm1 = nn.BatchNorm1d(int(channels))
            self.conv2 = CausalConv1d()
            self.norm2 = nn.BatchNorm1d(int(channels))
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(float(dropout))

        def forward(self, x):
            residual = x
            z = self.conv1(x)
            z = self.norm1(z)
            z = self.activation(z)
            z = self.dropout(z)
            z = self.conv2(z)
            z = self.norm2(z)
            z = self.dropout(z)
            return self.activation(z + residual)

    return ResidualTemporalBlock()


__all__ = ["build_residual_temporal_block"]
