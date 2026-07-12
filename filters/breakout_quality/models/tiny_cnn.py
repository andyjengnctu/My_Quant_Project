"""Legacy-compatible tiny 1D CNN for breakout quality classification."""

from __future__ import annotations


def build_tiny_cnn(nn, torch, *, feature_count: int, context_count: int):
    class TinyBreakoutQualityCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(int(feature_count), 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Conv1d(32, 32, kernel_size=5, padding=4, dilation=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(32 + int(context_count), 32),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(32, 2),
            )

        def forward(self, x, context):
            z = x.transpose(1, 2)
            z = self.conv(z).squeeze(-1)
            z = torch.cat([z, context], dim=1)
            return self.head(z)

    return TinyBreakoutQualityCNN()


__all__ = ["build_tiny_cnn"]
