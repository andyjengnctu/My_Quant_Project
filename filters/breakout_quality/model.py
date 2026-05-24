"""Tiny CNN model for breakout quality filter."""

from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise RuntimeError("breakout quality DL 訓練需要 PyTorch；請先安裝 torch") from exc
    return torch, nn


def build_model(feature_count: int, context_count: int):
    torch, nn = require_torch()

    class TinyBreakoutQualityCNN(nn.Module):
        def __init__(self, feature_count: int, context_count: int):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(feature_count, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(32 + context_count, 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 2),
            )

        def forward(self, x, context):
            z = x.transpose(1, 2)
            z = self.conv(z).squeeze(-1)
            z = torch.cat([z, context], dim=1)
            return self.head(z)

    return TinyBreakoutQualityCNN(int(feature_count), int(context_count))


__all__ = ["build_model", "require_torch"]
