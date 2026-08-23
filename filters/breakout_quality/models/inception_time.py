"""Single InceptionTime classifier for breakout-quality time-series inputs."""

from __future__ import annotations


def build_inception_time(nn, torch, *, feature_count: int, context_count: int, spec):
    use_risk_context = str(spec.architecture) == "inception_time_risk_context_v1"
    use_conditional_mfe_safety = (
        str(spec.architecture) == "inception_time_conditional_mfe_safety_v1"
    )
    if bool(spec.use_dataset_context) != bool(use_risk_context):
        raise ValueError("InceptionTime dataset context contract與architecture不一致")
    if use_risk_context and int(context_count) != 5:
        raise ValueError("MR-13J InceptionTime risk context固定需要5個universal geometry features")
    # Historical sequence-only InceptionTime callers may still pass the Dataset event-context
    # width through the generic factory.  The accepted 9A/13E architecture intentionally
    # ignores that tensor, so preserve the frozen checkpoint/API behavior instead of turning
    # an unused context_count into a new architecture contract.
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
            if use_risk_context:
                context_width = int(spec.head_width or 16)
                self.context_network = nn.Sequential(
                    nn.LayerNorm(int(context_count)),
                    nn.Linear(int(context_count), context_width),
                    nn.ReLU(),
                    nn.Linear(context_width, context_width),
                    nn.ReLU(),
                )
                classifier_input = module_output_channels + context_width
            else:
                self.context_network = None
                classifier_input = module_output_channels
            self.classifier = nn.Linear(classifier_input, 2)
            self.conditional_safety_classifier = (
                nn.Linear(module_output_channels + 1, 2)
                if use_conditional_mfe_safety
                else None
            )

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

        def _encoded_for_heads(self, x, context):
            encoded = self.dropout(self.encode(x))
            if self.context_network is not None:
                if context is None or context.ndim != 2 or int(context.shape[1]) != int(context_count):
                    raise ValueError("MR-13J risk context tensor shape不一致")
                return torch.cat([encoded, self.context_network(context)], dim=1), encoded
            return encoded, encoded

        def forward_conditional_heads(self, x, context):
            if self.conditional_safety_classifier is None:
                raise ValueError("目前architecture沒有conditional MFE-Safety heads")
            primary_input, shared_encoded = self._encoded_for_heads(x, context)
            primary_logits = self.classifier(primary_input)
            primary_probability = torch.softmax(primary_logits.float(), dim=1)[:, 1]
            conditional_context = primary_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
            conditional_logits = self.conditional_safety_classifier(
                torch.cat([shared_encoded, conditional_context], dim=1)
            )
            return primary_logits, conditional_logits

        def forward_output_head(self, x, context, output_head: str):
            head = str(output_head).strip().lower()
            if head in {"primary", "mfe", "primary_mfe"}:
                if self.conditional_safety_classifier is not None:
                    return self.forward_conditional_heads(x, context)[0]
                primary_input, _shared = self._encoded_for_heads(x, context)
                return self.classifier(primary_input)
            if head in {"conditional_safety", "safety"}:
                return self.forward_conditional_heads(x, context)[1]
            if head in {"conditional_both", "both"}:
                primary_logits, conditional_logits = self.forward_conditional_heads(x, context)
                return torch.cat([primary_logits, conditional_logits], dim=1)
            raise ValueError(f"未知InceptionTime output head: {output_head!r}")

        def forward(self, x, context):
            if self.conditional_safety_classifier is not None:
                return self.forward_conditional_heads(x, context)[0]
            primary_input, _shared = self._encoded_for_heads(x, context)
            return self.classifier(primary_input)

    return InceptionTimeClassifier()


__all__ = ["build_inception_time"]
