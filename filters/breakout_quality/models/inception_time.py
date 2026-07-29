"""Single InceptionTime classifier for breakout-quality time-series inputs."""

from __future__ import annotations


def build_inception_time(nn, torch, *, feature_count: int, context_count: int, spec):
    if bool(spec.use_dataset_context):
        raise ValueError("InceptionTime 9A 必須停用 Dataset event context")
    depth = int(spec.inception_depth)
    filters = int(spec.inception_filters)
    bottleneck_channels = int(spec.inception_bottleneck_channels)
    kernel_sizes = tuple(int(value) for value in spec.inception_kernel_sizes)
    residual_every = int(spec.inception_residual_every)
    if depth < 1 or filters < 1 or bottleneck_channels < 1 or residual_every < 1:
        raise ValueError("InceptionTime spec 必須使用正整數 depth／filters／bottleneck／residual interval")
    if not kernel_sizes or any(value < 1 or value % 2 == 0 for value in kernel_sizes):
        raise ValueError("InceptionTime kernels 必須是非空正奇數")

    normalization = str(spec.normalization or "batch_norm").strip().lower()
    normalization_groups = (
        None if spec.normalization_groups is None else int(spec.normalization_groups)
    )
    if normalization not in {"batch_norm", "group_norm"}:
        raise ValueError("InceptionTime normalization 只支援 batch_norm／group_norm")

    module_output_channels = filters * (len(kernel_sizes) + 1)
    if normalization == "group_norm":
        if normalization_groups is None or normalization_groups < 1:
            raise ValueError("InceptionTime GroupNorm 必須指定正整數 groups")
        if module_output_channels % normalization_groups != 0:
            raise ValueError("InceptionTime GroupNorm groups 必須整除輸出 channels")

    def build_normalization():
        if normalization == "group_norm":
            return nn.GroupNorm(normalization_groups, module_output_channels)
        return nn.BatchNorm1d(module_output_channels)

    class InceptionModule(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.bottleneck = nn.Conv1d(
                int(in_channels),
                bottleneck_channels,
                kernel_size=1,
                bias=False,
            )
            self.convolutions = nn.ModuleList(
                [
                    nn.Conv1d(
                        bottleneck_channels,
                        filters,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    )
                    for kernel_size in kernel_sizes
                ]
            )
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
            self.pool_projection = nn.Conv1d(
                int(in_channels), filters, kernel_size=1, bias=False
            )
            self.normalization = build_normalization()
            self.activation = nn.ReLU()

        def forward(self, x):
            bottleneck = self.bottleneck(x)
            branches = [conv(bottleneck) for conv in self.convolutions]
            branches.append(self.pool_projection(self.pool(x)))
            return self.activation(self.normalization(torch.cat(branches, dim=1)))

    class ResidualProjection(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(
                    int(in_channels),
                    module_output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                build_normalization(),
            )

        def forward(self, x):
            return self.network(x)

    class InceptionTimeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            modules = []
            shortcuts = []
            in_channels = int(feature_count)
            residual_channels = in_channels
            for module_index in range(depth):
                modules.append(InceptionModule(in_channels))
                in_channels = module_output_channels
                if (module_index + 1) % residual_every == 0:
                    shortcuts.append(ResidualProjection(residual_channels))
                    residual_channels = module_output_channels
            if depth % residual_every != 0:
                raise ValueError("InceptionTime depth 必須可被 residual interval 整除")
            self.inception_modules = nn.ModuleList(modules)
            self.residual_projections = nn.ModuleList(shortcuts)
            self.residual_activation = nn.ReLU()
            self.dropout = nn.Dropout(float(spec.dropout))
            self.classifier = nn.Linear(module_output_channels, 2)

        def encode(self, x):
            z = x.transpose(1, 2)
            residual = z
            shortcut_index = 0
            for module_index, module in enumerate(self.inception_modules, start=1):
                z = module(z)
                if module_index % residual_every == 0:
                    z = self.residual_activation(
                        z + self.residual_projections[shortcut_index](residual)
                    )
                    residual = z
                    shortcut_index += 1
            return torch.mean(z, dim=2)

        def forward(self, x, context):
            del context
            return self.classifier(self.dropout(self.encode(x)))

    return InceptionTimeClassifier()


__all__ = ["build_inception_time"]
