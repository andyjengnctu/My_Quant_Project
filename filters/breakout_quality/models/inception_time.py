"""Current InceptionTime single-ranker and shared Safety/MFE architectures."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import get_architecture_descriptor


def build_inception_time(nn, torch, *, feature_count: int, context_count: int, spec):
    descriptor = get_architecture_descriptor(spec.architecture)
    use_shared_safety_mfe = descriptor.has_capability("shared_safety_mfe")
    if bool(spec.use_dataset_context):
        raise ValueError("Current InceptionTime models require sequence-only inputs")
    depth = int(spec.inception_depth)
    filters = int(spec.inception_filters)
    bottleneck_channels = int(spec.inception_bottleneck_channels)
    kernel_sizes = tuple(int(value) for value in spec.inception_kernel_sizes)
    module_dilations = tuple(int(value) for value in (spec.inception_module_dilations or (1,) * depth))
    residual_every = int(spec.inception_residual_every)
    if depth < 1 or filters < 1 or bottleneck_channels < 1 or residual_every < 1:
        raise ValueError("InceptionTime spec requires positive depth, widths and residual interval")
    if not kernel_sizes or any(value < 1 or value % 2 == 0 for value in kernel_sizes):
        raise ValueError("InceptionTime kernels must be non-empty positive odd integers")
    if len(module_dilations) != depth or any(value < 1 for value in module_dilations):
        raise ValueError("InceptionTime module dilations must match depth")
    module_output_channels = filters * (len(kernel_sizes) + 1)

    def build_normalization():
        return nn.BatchNorm1d(module_output_channels)

    class InceptionModule(nn.Module):
        def __init__(self, in_channels: int, *, dilation: int):
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
                        dilation=int(dilation),
                        padding=int(dilation) * (kernel_size // 2),
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
                modules.append(InceptionModule(in_channels, dilation=module_dilations[module_index]))
                in_channels = module_output_channels
                if (module_index + 1) % residual_every == 0:
                    shortcuts.append(ResidualProjection(residual_channels))
                    residual_channels = module_output_channels
            self.inception_modules = nn.ModuleList(modules)
            self.residual_projections = nn.ModuleList(shortcuts)
            self.residual_activation = nn.ReLU()
            self.dropout = nn.Dropout(float(spec.dropout))
            self.default_output_width = int(spec.classifier_output_classes or 2)
            # Preserve creation order and names for existing checkpoint tensors and seeds.
            self.classifier = nn.Linear(module_output_channels, self.default_output_width)
            self.raw_safety_classifier = nn.Linear(module_output_channels, 2) if use_shared_safety_mfe else None
            self.raw_mfe_classifier = nn.Linear(module_output_channels, 2) if use_shared_safety_mfe else None

        def _run_residual_stack(self, z, modules, projections):
            residual = z
            shortcut_index = 0
            for module_index, module in enumerate(modules, start=1):
                z = module(z)
                if module_index % residual_every == 0:
                    z = self.residual_activation(
                        z + projections[shortcut_index](residual)
                    )
                    residual = z
                    shortcut_index += 1
            return z


        def encode_feature_map(self, x):
            return self._run_residual_stack(x.transpose(1, 2), self.inception_modules, self.residual_projections)

        def encode(self, x):
            z = self.encode_feature_map(x)
            return torch.mean(z, dim=2)


        @property
        def encoder_embedding_width(self):
            return int(module_output_channels)


        def encoder_parameters(self):
            return tuple(self.inception_modules.parameters()) + tuple(
                self.residual_projections.parameters()
            )


        def export_encoder_state_dict(self):
            state = {}
            for owner_name in ("inception_modules", "residual_projections"):
                owner = getattr(self, owner_name)
                for key, value in owner.state_dict().items():
                    state[f"{owner_name}.{key}"] = value.detach().cpu().clone()
            return state


        def load_encoder_state_dict(self, state):
            supplied = {str(key): value for key, value in dict(state).items()}
            expected = set(self.export_encoder_state_dict())
            if set(supplied) != expected:
                missing = sorted(expected.difference(supplied))
                extra = sorted(set(supplied).difference(expected))
                raise ValueError(
                    "pretrained encoder state keys不符合AO shared encoder contract: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            for owner_name in ("inception_modules", "residual_projections"):
                prefix = f"{owner_name}."
                owner_state = {
                    key[len(prefix):]: value
                    for key, value in supplied.items()
                    if key.startswith(prefix)
                }
                getattr(self, owner_name).load_state_dict(owner_state, strict=True)


        def _encoded_for_heads(self, x, context):
            # Event-context tensors remain accepted but unused by the sequence-only models.
            encoded = self.dropout(self.encode(x))
            return encoded, encoded

        def forward_safety_mfe_heads(self, x, context, *, relation_history_x=None, include_adverse_quantiles=False):
            if self.raw_safety_classifier is None or self.raw_mfe_classifier is None:
                raise ValueError("Current architecture has no shared Safety/MFE heads")
            if include_adverse_quantiles:
                raise ValueError("Adverse-quantile auxiliary architecture is retired")
            _primary, encoded = self._encoded_for_heads(x, context)
            return self.raw_safety_classifier(encoded), self.raw_mfe_classifier(encoded)

        def forward_output_head(self, x, context, output_head: str, *, relation_history_x=None):
            head = str(output_head).strip().lower()
            if head in {"primary", "mfe", "primary_mfe"}:
                return self.forward(x, context)
            if head in {"conditional_both", "both"}:
                safety, mfe = self.forward_safety_mfe_heads(x, context)
                return torch.cat([safety, mfe], dim=1)
            if head in {"raw_safety", "safety_condition"}:
                return self.forward_safety_mfe_heads(x, context)[0]
            if head in {"conditional_mfe", "raw_mfe", "final"}:
                return self.forward_safety_mfe_heads(x, context)[1]
            raise ValueError(f"Unsupported InceptionTime output head: {output_head!r}")

        def forward(self, x, context, *, relation_history_x=None):
            if self.raw_mfe_classifier is not None:
                return self.forward_safety_mfe_heads(x, context)[1]
            primary_input, _shared = self._encoded_for_heads(x, context)
            return self.classifier(primary_input)

    return InceptionTimeClassifier()


__all__ = ["build_inception_time"]
