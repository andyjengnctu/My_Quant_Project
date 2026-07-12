"""Medium residual dilated TCN for long-horizon breakout patterns."""

from __future__ import annotations

from filters.breakout_quality.models.blocks import build_residual_temporal_block


def build_residual_tcn(nn, torch, *, feature_count: int, context_count: int, spec):
    class ResidualBreakoutQualityTCN(nn.Module):
        def __init__(self):
            super().__init__()
            channels = int(spec.channels)
            self.stem = nn.Sequential(
                nn.Conv1d(int(feature_count), channels, kernel_size=1),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            )
            self.blocks = nn.ModuleList(
                [
                    build_residual_temporal_block(
                        nn,
                        torch,
                        channels=channels,
                        kernel_size=int(spec.kernel_size),
                        dilation=int(dilation),
                        dropout=float(spec.dropout),
                    )
                    for dilation in spec.dilations
                ]
            )
            pooled_width = channels * len(spec.pooling)
            self.head = nn.Sequential(
                nn.Linear(pooled_width + int(context_count), 64),
                nn.ReLU(),
                nn.Dropout(float(spec.dropout)),
                nn.Linear(64, 2),
            )

        def forward(self, x, context):
            z = self.stem(x.transpose(1, 2))
            for block in self.blocks:
                z = block(z)
            pooled = []
            for pooling_name in spec.pooling:
                if pooling_name == "last":
                    pooled.append(z[:, :, -1])
                elif pooling_name == "average":
                    pooled.append(torch.mean(z, dim=2))
                elif pooling_name == "max":
                    pooled.append(torch.amax(z, dim=2))
                else:
                    raise RuntimeError(f"未知 pooling: {pooling_name}")
            summary = torch.cat(pooled, dim=1)
            return self.head(torch.cat([summary, context], dim=1))

    return ResidualBreakoutQualityTCN()


__all__ = ["build_residual_tcn"]
