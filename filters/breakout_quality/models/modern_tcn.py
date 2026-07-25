"""ModernTCN-style classifier for breakout-quality time-series inputs."""

from __future__ import annotations


def build_modern_tcn(nn, torch, *, feature_count: int, context_count: int, spec):
    """Build the project-specific single ModernTCN classifier.

    The implementation isolates the intended architecture change: large-kernel
    depthwise temporal convolution followed by pointwise channel mixing inside
    residual blocks. Dataset event context remains disabled.
    """

    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("ModernTCN 9B 必須停用 Dataset event context")

    depth = int(spec.modern_tcn_depth)
    channels = int(spec.modern_tcn_channels)
    kernel_size = int(spec.modern_tcn_kernel_size)
    expansion_ratio = int(spec.modern_tcn_expansion_ratio)
    dropout = float(spec.dropout)
    if depth < 1 or channels < 1 or expansion_ratio < 1:
        raise ValueError("ModernTCN spec 必須使用正整數 depth／channels／expansion ratio")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("ModernTCN large kernel 必須是 >=3 的奇數")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("ModernTCN dropout 必須介於 [0, 1)")

    expanded_channels = channels * expansion_ratio

    class ModernTCNBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.depthwise = nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            )
            self.temporal_norm = nn.BatchNorm1d(channels)
            self.pointwise_expand = nn.Conv1d(
                channels,
                expanded_channels,
                kernel_size=1,
                bias=False,
            )
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dropout)
            self.pointwise_project = nn.Conv1d(
                expanded_channels,
                channels,
                kernel_size=1,
                bias=False,
            )
            self.output_norm = nn.BatchNorm1d(channels)
            self.residual_activation = nn.GELU()

        def forward(self, x):
            residual = x
            z = self.depthwise(x)
            z = self.temporal_norm(z)
            z = self.pointwise_expand(z)
            z = self.activation(z)
            z = self.dropout(z)
            z = self.pointwise_project(z)
            z = self.output_norm(z)
            return self.residual_activation(residual + z)

    class ModernTCNClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(
                    int(feature_count),
                    channels,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList([ModernTCNBlock() for _ in range(depth)])
            self.head_dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(channels, 2)

        def forward(self, x, context):
            del context
            z = self.stem(x.transpose(1, 2))
            for block in self.blocks:
                z = block(z)
            pooled = torch.mean(z, dim=2)
            return self.classifier(self.head_dropout(pooled))

    return ModernTCNClassifier()


__all__ = ["build_modern_tcn"]
