"""Single InceptionTime classifier for breakout-quality time-series inputs."""

from __future__ import annotations

from filters.breakout_quality.models.architectures import get_architecture_descriptor


def build_inception_time(nn, torch, *, feature_count: int, context_count: int, spec):
    descriptor = get_architecture_descriptor(spec.architecture)
    use_risk_context = descriptor.has_capability("risk_context")
    use_predicted_upside_context = descriptor.has_capability("predicted_upside_context")
    use_predicted_safety_context = descriptor.has_capability("predicted_safety_context")
    use_predicted_scalar_context = bool(
        use_predicted_upside_context or use_predicted_safety_context
    )
    use_conditional_mfe_safety = descriptor.has_capability("conditional_mfe_safety")
    use_safety_conditional_mfe = descriptor.has_capability("safety_conditional_mfe")
    use_shared_safety_mfe = descriptor.has_capability("shared_safety_mfe")
    use_same_date_dynamic_hypergraph_safety = descriptor.has_capability(
        "same_date_dynamic_hypergraph_safety"
    )
    use_same_date_dynamic_hypergraph_relation_change_safety = descriptor.has_capability(
        "same_date_dynamic_hypergraph_relation_change_safety"
    )
    use_task_specific_safety_mfe = descriptor.has_capability("task_specific_safety_mfe")
    use_safety_attention_pool = descriptor.has_capability("safety_attention_pool")
    use_safety_temporal_self_attention = descriptor.has_capability("safety_temporal_self_attention")
    use_safety_pairwise_temporal_relation_bias = descriptor.has_capability(
        "safety_pairwise_temporal_relation_bias"
    )
    use_price_volume_structure_safety = descriptor.has_capability("price_volume_structure_safety")
    use_price_volume_local_structure_safety = descriptor.has_capability(
        "price_volume_local_structure_safety"
    )
    use_price_volume_multiscale_local_safety = descriptor.has_capability(
        "price_volume_multiscale_local_safety"
    )
    use_price_volume_position_aware_multiscale_safety = descriptor.has_capability(
        "price_volume_position_aware_multiscale_safety"
    )
    if (use_price_volume_local_structure_safety or use_price_volume_multiscale_local_safety) and not use_price_volume_structure_safety:
        raise ValueError("Local Price-Volume structure必須建立在global Price-Volume structure上")
    if sum(
        int(flag)
        for flag in (
            use_price_volume_local_structure_safety,
            use_price_volume_multiscale_local_safety,
            use_price_volume_position_aware_multiscale_safety,
        )
    ) > 1:
        raise ValueError("Price-Volume local/position-aware treatment不可同時啟用")
    if use_safety_attention_pool and use_safety_temporal_self_attention:
        raise ValueError("Safety scalar pooling與temporal self-attention不可同時啟用")
    if use_safety_pairwise_temporal_relation_bias and not use_safety_temporal_self_attention:
        raise ValueError("Pairwise temporal relation bias必須建立在Safety temporal self-attention上")
    if use_same_date_dynamic_hypergraph_safety and not use_shared_safety_mfe:
        raise ValueError("same-date dynamic hypergraph Safety residual必須建立在shared Safety/MFE architecture上")
    if use_same_date_dynamic_hypergraph_relation_change_safety and not use_same_date_dynamic_hypergraph_safety:
        raise ValueError("relation-change hypergraph Safety residual必須建立在same-date dynamic hypergraph上")
    use_safety_raw_mfe_hmhs = descriptor.has_capability("safety_raw_mfe_hmhs")
    use_nonlinear_hmhs_head = descriptor.has_capability("nonlinear_hmhs_head")
    use_joint_attention_pool = descriptor.has_capability("joint_attention_pool")
    if bool(spec.use_dataset_context) != bool(use_risk_context or use_predicted_scalar_context):
        raise ValueError("InceptionTime dataset context contract與architecture不一致")
    if use_risk_context and int(context_count) != 5:
        raise ValueError("MR-13J InceptionTime risk context固定需要5個universal geometry features")
    if use_predicted_scalar_context and int(context_count) != 1:
        raise ValueError("MR-13AC/AD predicted context固定需要1個PIT-safe scalar")
    # Historical sequence-only InceptionTime callers may still pass the Dataset event-context
    # width through the generic factory.  The accepted 9A/13E architecture intentionally
    # ignores that tensor, so preserve the frozen checkpoint/API behavior instead of turning
    # an unused context_count into a new architecture contract.
    depth = int(spec.inception_depth)
    filters = int(spec.inception_filters)
    bottleneck_channels = int(spec.inception_bottleneck_channels)
    kernel_sizes = tuple(int(value) for value in spec.inception_kernel_sizes)
    module_dilations = tuple(int(value) for value in (spec.inception_module_dilations or (1,) * depth))
    residual_every = int(spec.inception_residual_every)
    if depth < 1 or filters < 1 or bottleneck_channels < 1 or residual_every < 1:
        raise ValueError("InceptionTime spec 必須使用正整數 depth／filters／bottleneck／residual interval")
    if not kernel_sizes or any(value < 1 or value % 2 == 0 for value in kernel_sizes):
        raise ValueError("InceptionTime kernels 必須是非空正奇數")
    if len(module_dilations) != depth or any(value < 1 for value in module_dilations):
        raise ValueError("InceptionTime module dilations必須與depth同長且皆為正整數")

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
            if depth % residual_every != 0:
                raise ValueError("InceptionTime depth 必須可被 residual interval 整除")
            shared_depth = depth
            if use_task_specific_safety_mfe:
                shared_depth = depth - residual_every
                if shared_depth < residual_every or shared_depth % residual_every != 0:
                    raise ValueError(
                        "Task-specific InceptionTime 需要至少一個完整shared residual group與一個完整task-specific residual group"
                    )

            modules = []
            shortcuts = []
            in_channels = int(feature_count)
            residual_channels = in_channels
            for module_index in range(shared_depth):
                modules.append(
                    InceptionModule(in_channels, dilation=module_dilations[module_index])
                )
                in_channels = module_output_channels
                if (module_index + 1) % residual_every == 0:
                    shortcuts.append(ResidualProjection(residual_channels))
                    residual_channels = module_output_channels
            self.inception_modules = nn.ModuleList(modules)
            self.residual_projections = nn.ModuleList(shortcuts)

            def build_task_specific_group():
                branch_modules = nn.ModuleList(
                    [
                        InceptionModule(module_output_channels, dilation=module_dilations[shared_depth + offset])
                        for offset in range(residual_every)
                    ]
                )
                branch_shortcuts = nn.ModuleList([ResidualProjection(module_output_channels)])
                return branch_modules, branch_shortcuts

            if use_task_specific_safety_mfe:
                (
                    self.safety_inception_modules,
                    self.safety_residual_projections,
                ) = build_task_specific_group()
                (
                    self.mfe_inception_modules,
                    self.mfe_residual_projections,
                ) = build_task_specific_group()
            else:
                self.safety_inception_modules = nn.ModuleList()
                self.safety_residual_projections = nn.ModuleList()
                self.mfe_inception_modules = nn.ModuleList()
                self.mfe_residual_projections = nn.ModuleList()
            self.residual_activation = nn.ReLU()
            self.dropout = nn.Dropout(float(spec.dropout))
            self.direct_context_concat = bool(use_predicted_scalar_context)
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
            elif use_predicted_scalar_context:
                self.context_network = None
                classifier_input = module_output_channels + 1
            else:
                self.context_network = None
                classifier_input = module_output_channels
            self.classifier = nn.Linear(classifier_input, 2)
            self.conditional_safety_classifier = (
                nn.Linear(module_output_channels + 1, 2)
                if use_conditional_mfe_safety
                else None
            )
            self.raw_safety_classifier = (
                nn.Linear(module_output_channels, 2)
                if (
                    use_safety_conditional_mfe
                    or use_shared_safety_mfe
                    or use_task_specific_safety_mfe
                    or use_safety_raw_mfe_hmhs
                )
                else None
            )
            self.raw_mfe_classifier = (
                nn.Linear(module_output_channels, 2)
                if use_shared_safety_mfe or use_task_specific_safety_mfe
                else None
            )
            self.conditional_mfe_classifier = (
                nn.Linear(module_output_channels + 1, 2)
                if use_safety_conditional_mfe or use_safety_raw_mfe_hmhs
                else None
            )
            if use_nonlinear_hmhs_head:
                hidden_width = int(spec.head_width or module_output_channels)
                if hidden_width != module_output_channels:
                    raise ValueError(
                        "MR-13V Direct HM/HS MLP hidden width必須等於shared latent width"
                    )
                self.joint_hmhs_classifier = nn.Sequential(
                    nn.Linear(module_output_channels, hidden_width),
                    nn.ReLU(),
                    nn.Linear(hidden_width, 2),
                )
            else:
                self.joint_hmhs_classifier = (
                    nn.Linear(module_output_channels, 2)
                    if use_safety_raw_mfe_hmhs
                    else None
                )
            # Keep this parameterized pooling module after all MR-13W modules so
            # the same seed initializes every shared/marginal/MLP parameter
            # identically in the strict MR-13W -> MR-13X architecture contrast.
            self.joint_attention_scorer = (
                nn.Conv1d(module_output_channels, 1, kernel_size=1, bias=True)
                if use_joint_attention_pool
                else None
            )
            # Safety-specific temporal pooling is deliberately a single scalar 1x1 scorer.
            # It adds no attention-head count, hidden width, temperature or query-count knob.
            # Keep it after all AO-shared parameters so same-seed AO/AW common parameter
            # initialization remains identical; only this scorer is newly initialized.
            self.safety_attention_scorer = (
                nn.Conv1d(module_output_channels, 1, kernel_size=1, bias=True)
                if use_safety_attention_pool
                else None
            )
            # Safety temporal interaction is deliberately a minimal single-head block:
            # Q/K/V keep the canonical channel width, scaled-dot-product attention uses
            # PyTorch's canonical 1/sqrt(C) scale, and the context is added residually
            # before the unchanged GAP.  No FFN/positional encoding/dropout/window/heads
            # are introduced.  Parameters are appended after AO-common modules so the
            # same seed preserves every AO parameter and the complete MFE path exactly.
            self.safety_temporal_query = (
                nn.Conv1d(module_output_channels, module_output_channels, kernel_size=1, bias=False)
                if use_safety_temporal_self_attention
                else None
            )
            self.safety_temporal_key = (
                nn.Conv1d(module_output_channels, module_output_channels, kernel_size=1, bias=False)
                if use_safety_temporal_self_attention
                else None
            )
            self.safety_temporal_value = (
                nn.Conv1d(module_output_channels, module_output_channels, kernel_size=1, bias=False)
                if use_safety_temporal_self_attention
                else None
            )
            if use_safety_pairwise_temporal_relation_bias:
                from filters.breakout_quality.models.pairwise_temporal_relation import (
                    build_pairwise_temporal_relation_bias,
                )

                self.safety_temporal_relation_bias = build_pairwise_temporal_relation_bias(
                    nn,
                    torch,
                    feature_count=int(feature_count),
                    spec=spec,
                )
            else:
                self.safety_temporal_relation_bias = None
            if use_price_volume_structure_safety:
                from filters.breakout_quality.models.price_volume_structure import (
                    build_price_volume_local_structure_encoder,
                    build_price_volume_multiscale_local_encoder,
                    build_price_volume_structure_encoder,
                )

                self.price_volume_structure_encoder = build_price_volume_structure_encoder(
                    nn,
                    torch,
                    feature_count=int(feature_count),
                    output_width=module_output_channels,
                    spec=spec,
                )
                self.price_volume_local_structure_encoder = (
                    build_price_volume_local_structure_encoder(
                        nn,
                        torch,
                        output_width=module_output_channels,
                        spec=spec,
                    )
                    if use_price_volume_local_structure_safety
                    else None
                )
                self.price_volume_multiscale_local_encoder = (
                    build_price_volume_multiscale_local_encoder(
                        nn,
                        torch,
                        feature_count=int(feature_count),
                        output_width=module_output_channels,
                        spec=spec,
                    )
                    if use_price_volume_multiscale_local_safety
                    else None
                )
            else:
                self.price_volume_structure_encoder = None
                self.price_volume_local_structure_encoder = None
                self.price_volume_multiscale_local_encoder = None
            if use_price_volume_position_aware_multiscale_safety:
                from filters.breakout_quality.models.price_volume_structure import (
                    build_price_volume_position_aware_multiscale_encoder,
                )

                self.price_volume_position_aware_multiscale_encoder = (
                    build_price_volume_position_aware_multiscale_encoder(
                        nn,
                        torch,
                        feature_count=int(feature_count),
                        output_width=module_output_channels,
                        spec=spec,
                    )
                )
            else:
                self.price_volume_position_aware_multiscale_encoder = None
            if use_same_date_dynamic_hypergraph_safety:
                from filters.breakout_quality.models.dynamic_hypergraph import (
                    build_same_date_dynamic_hypergraph_safety_residual,
                )

                self.same_date_dynamic_hypergraph_safety_residual = (
                    build_same_date_dynamic_hypergraph_safety_residual(
                        nn,
                        torch,
                        latent_width=module_output_channels,
                        spec=spec,
                    )
                )
            else:
                self.same_date_dynamic_hypergraph_safety_residual = None
            self.requires_same_date_relations = bool(
                use_same_date_dynamic_hypergraph_safety
            )
            self.same_date_relation_history_steps = int(
                spec.same_date_relation_history_steps or 0
            )
            if bool(use_same_date_dynamic_hypergraph_relation_change_safety) != bool(
                self.same_date_relation_history_steps == 1
            ):
                raise ValueError("relation-change hypergraph capability與history-step contract不一致")

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
            return self._run_residual_stack(
                x.transpose(1, 2), self.inception_modules, self.residual_projections
            )

        def encode_task_specific_feature_maps(self, x):
            if not use_task_specific_safety_mfe:
                raise ValueError("目前architecture沒有task-specific Safety/MFE representation")
            shared = self.encode_feature_map(x)
            safety = self._run_residual_stack(
                shared, self.safety_inception_modules, self.safety_residual_projections
            )
            mfe = self._run_residual_stack(
                shared, self.mfe_inception_modules, self.mfe_residual_projections
            )
            return safety, mfe

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

        def _encode_relation_view_without_state_update(self, x):
            """Encode relational-only inputs without gradient or BatchNorm state mutation."""

            owners = (self.inception_modules, self.residual_projections)
            training_states = tuple(bool(owner.training) for owner in owners)
            try:
                for owner in owners:
                    owner.eval()
                with torch.no_grad():
                    return self.encode(x).detach()
            finally:
                for owner, was_training in zip(owners, training_states):
                    owner.train(was_training)

        def safety_attention_weights(self, x):
            if self.safety_attention_scorer is None:
                raise ValueError("目前architecture沒有Safety temporal attention pooling")
            if use_task_specific_safety_mfe:
                feature_map, _mfe_map = self.encode_task_specific_feature_maps(x)
            else:
                feature_map = self.encode_feature_map(x)
            logits = self.safety_attention_scorer(feature_map).squeeze(1)
            return torch.softmax(logits.float(), dim=1).to(feature_map.dtype)

        def _safety_attention_pool(self, feature_map):
            if self.safety_attention_scorer is None:
                return torch.mean(feature_map, dim=2)
            logits = self.safety_attention_scorer(feature_map).squeeze(1)
            weights = torch.softmax(logits.float(), dim=1).to(feature_map.dtype)
            return torch.sum(feature_map * weights.unsqueeze(1), dim=2)

        def _safety_temporal_interaction(self, feature_map, sequence_input=None):
            if (
                self.safety_temporal_query is None
                or self.safety_temporal_key is None
                or self.safety_temporal_value is None
            ):
                return feature_map
            query = self.safety_temporal_query(feature_map).transpose(1, 2).unsqueeze(1)
            key = self.safety_temporal_key(feature_map).transpose(1, 2).unsqueeze(1)
            value = self.safety_temporal_value(feature_map).transpose(1, 2).unsqueeze(1)
            if self.safety_temporal_relation_bias is not None:
                if sequence_input is None:
                    raise ValueError("Pairwise temporal relation bias需要原始sequence input")
                context = self.safety_temporal_relation_bias.scaled_dot_product_attention(
                    query, key, value, sequence_input
                )
            else:
                context = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                )
            return feature_map + context.squeeze(1).transpose(1, 2).to(feature_map.dtype)

        def joint_attention_weights(self, x):
            if self.joint_attention_scorer is None:
                raise ValueError("目前architecture沒有Joint temporal attention pooling")
            feature_map = self.encode_feature_map(x)
            logits = self.joint_attention_scorer(feature_map).squeeze(1)
            return torch.softmax(logits.float(), dim=1).to(feature_map.dtype)

        def _joint_attention_pool(self, feature_map):
            if self.joint_attention_scorer is None:
                return torch.mean(feature_map, dim=2)
            logits = self.joint_attention_scorer(feature_map).squeeze(1)
            weights = torch.softmax(logits.float(), dim=1).to(feature_map.dtype)
            return torch.sum(feature_map * weights.unsqueeze(1), dim=2)

        def _encoded_for_heads(self, x, context):
            encoded = self.dropout(self.encode(x))
            if self.context_network is not None:
                if context is None or context.ndim != 2 or int(context.shape[1]) != int(context_count):
                    raise ValueError("MR-13J risk context tensor shape不一致")
                return torch.cat([encoded, self.context_network(context)], dim=1), encoded
            if self.direct_context_concat:
                if context is None or context.ndim != 2 or int(context.shape[1]) != 1:
                    raise ValueError("MR-13AC/AD predicted context tensor shape不一致")
                if not bool(torch.isfinite(context).all()):
                    raise ValueError("MR-13AC/AD predicted context不得含non-finite value")
                return torch.cat([encoded, context.to(encoded.dtype)], dim=1), encoded
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

        def forward_safety_mfe_heads(self, x, context, *, relation_history_x=None):
            """Return the canonical Raw-Safety + final-MFE pair for duo-head training.

            The final MFE topology is owned by the model spec: an independent raw-MFE
            head reads only the shared latent, while a Safety-conditioned MFE head also
            receives stop-gradient Safety probability.  The trainer consumes this
            capability and never needs to recognize the architecture identity.
            """
            if self.raw_safety_classifier is None:
                raise ValueError("目前architecture沒有Raw Safety head")
            if self.same_date_dynamic_hypergraph_safety_residual is not None:
                _primary_input, shared_encoded = self._encoded_for_heads(x, context)
                safety_logits = self.raw_safety_classifier(shared_encoded)
                relation_kwargs = {}
                if int(self.same_date_relation_history_steps) == 1:
                    if relation_history_x is None:
                        raise ValueError("relation-change hypergraph缺少previous-trading-date feature tensor")
                    relation_current_latent = self._encode_relation_view_without_state_update(x)
                    if int(relation_history_x.shape[0]) == 0:
                        previous_relation_latent = relation_current_latent.new_empty(
                            (0, int(self.encoder_embedding_width))
                        )
                    else:
                        previous_relation_latent = self._encode_relation_view_without_state_update(
                            relation_history_x
                        )
                    relation_kwargs = {
                        "relation_current_latent": relation_current_latent,
                        "previous_relation_latent": previous_relation_latent,
                    }
                elif relation_history_x is not None:
                    raise ValueError("same-date state-only hypergraph不得收到relation-history feature tensor")
                safety_logits = safety_logits + self.same_date_dynamic_hypergraph_safety_residual(
                    shared_encoded, **relation_kwargs
                ).to(safety_logits.dtype)
                if self.raw_mfe_classifier is None:
                    raise ValueError("Dynamic-hypergraph architecture沒有Raw MFE head")
                mfe_logits = self.raw_mfe_classifier(shared_encoded)
                return safety_logits, mfe_logits
            if use_task_specific_safety_mfe:
                safety_map, mfe_map = self.encode_task_specific_feature_maps(x)
                safety_pooled = (
                    self._safety_attention_pool(safety_map)
                    if use_safety_attention_pool
                    else torch.mean(safety_map, dim=2)
                )
                safety_encoded = self.dropout(safety_pooled)
                mfe_encoded = self.dropout(torch.mean(mfe_map, dim=2))
                safety_logits = self.raw_safety_classifier(safety_encoded)
                if self.raw_mfe_classifier is None:
                    raise ValueError("Task-specific architecture沒有Raw MFE head")
                mfe_logits = self.raw_mfe_classifier(mfe_encoded)
                return safety_logits, mfe_logits
            if use_safety_attention_pool or use_safety_temporal_self_attention:
                feature_map = self.encode_feature_map(x)
                mfe_pooled = torch.mean(feature_map, dim=2)
                if use_safety_attention_pool:
                    safety_pooled = self._safety_attention_pool(feature_map)
                else:
                    safety_pooled = torch.mean(
                        self._safety_temporal_interaction(feature_map, x), dim=2
                    )
                # Preserve AO's single dropout RNG draw on the MFE path and reuse the same
                # mask for Safety.  This keeps the controlled contrast focused on the
                # Safety representation primitive rather than a second stochastic mask.
                dropout_mask = self.dropout(torch.ones_like(mfe_pooled))
                mfe_encoded = mfe_pooled * dropout_mask
                safety_encoded = safety_pooled * dropout_mask
                safety_logits = self.raw_safety_classifier(safety_encoded)
                if self.raw_mfe_classifier is None:
                    raise ValueError("Safety-representation architecture沒有Raw MFE head")
                mfe_logits = self.raw_mfe_classifier(mfe_encoded)
                return safety_logits, mfe_logits
            _primary_input, shared_encoded = self._encoded_for_heads(x, context)
            if self.price_volume_position_aware_multiscale_encoder is not None:
                position_aware_residual = (
                    self.price_volume_position_aware_multiscale_encoder(x).to(
                        shared_encoded.dtype
                    )
                )
                safety_encoded = shared_encoded + position_aware_residual
            elif self.price_volume_structure_encoder is not None:
                if self.price_volume_local_structure_encoder is not None:
                    geometry_map, vap = (
                        self.price_volume_structure_encoder.build_structure_inputs(x)
                    )
                    structure_residual = (
                        self.price_volume_structure_encoder.encode_structure_inputs(
                            geometry_map, vap
                        ).to(shared_encoded.dtype)
                    )
                    globally_conditioned_safety = shared_encoded + structure_residual
                    local_structure_residual = (
                        self.price_volume_local_structure_encoder(
                            geometry_map,
                            vap,
                            globally_conditioned_safety,
                        ).to(shared_encoded.dtype)
                    )
                    safety_encoded = globally_conditioned_safety + local_structure_residual
                elif self.price_volume_multiscale_local_encoder is not None:
                    geometry_map, vap = (
                        self.price_volume_structure_encoder.build_structure_inputs(x)
                    )
                    structure_residual = (
                        self.price_volume_structure_encoder.encode_structure_inputs(
                            geometry_map, vap
                        ).to(shared_encoded.dtype)
                    )
                    local_multiscale_residual = (
                        self.price_volume_multiscale_local_encoder(
                            x, structure_residual
                        ).to(shared_encoded.dtype)
                    )
                    safety_encoded = (
                        shared_encoded + structure_residual + local_multiscale_residual
                    )
                else:
                    structure_residual = self.price_volume_structure_encoder(x).to(
                        shared_encoded.dtype
                    )
                    safety_encoded = shared_encoded + structure_residual
            else:
                safety_encoded = shared_encoded
            safety_logits = self.raw_safety_classifier(safety_encoded)
            if self.raw_mfe_classifier is not None:
                mfe_logits = self.raw_mfe_classifier(shared_encoded)
            elif self.conditional_mfe_classifier is not None:
                safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
                safety_context = safety_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
                mfe_logits = self.conditional_mfe_classifier(
                    torch.cat([shared_encoded, safety_context], dim=1)
                )
            else:
                raise ValueError("目前architecture沒有可用的final MFE head")
            return safety_logits, mfe_logits

        def forward_safety_conditional_mfe_heads(self, x, context):
            if self.raw_safety_classifier is None or self.conditional_mfe_classifier is None:
                raise ValueError("目前architecture沒有Safety→Conditional-MFE heads")
            _primary_input, shared_encoded = self._encoded_for_heads(x, context)
            safety_logits = self.raw_safety_classifier(shared_encoded)
            safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
            safety_context = safety_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
            conditional_mfe_logits = self.conditional_mfe_classifier(
                torch.cat([shared_encoded, safety_context], dim=1)
            )
            return safety_logits, conditional_mfe_logits

        def forward_safety_raw_mfe_hmhs_heads(self, x, context):
            if (
                self.raw_safety_classifier is None
                or self.conditional_mfe_classifier is None
                or self.joint_hmhs_classifier is None
            ):
                raise ValueError("目前architecture沒有Safety/Raw-MFE/HMHS tri-heads")
            if self.joint_attention_scorer is not None:
                feature_map = self.encode_feature_map(x)
                shared_encoded = self.dropout(torch.mean(feature_map, dim=2))
                joint_encoded = self.dropout(self._joint_attention_pool(feature_map))
            else:
                _primary_input, shared_encoded = self._encoded_for_heads(x, context)
                joint_encoded = shared_encoded
            safety_logits = self.raw_safety_classifier(shared_encoded)
            safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
            safety_context = safety_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
            raw_mfe_logits = self.conditional_mfe_classifier(
                torch.cat([shared_encoded, safety_context], dim=1)
            )
            joint_hmhs_logits = self.joint_hmhs_classifier(joint_encoded)
            return safety_logits, raw_mfe_logits, joint_hmhs_logits

        def forward_output_head(self, x, context, output_head: str, *, relation_history_x=None):
            head = str(output_head).strip().lower()
            if head in {"primary", "mfe", "primary_mfe"}:
                if self.conditional_safety_classifier is not None:
                    return self.forward_conditional_heads(x, context)[0]
                if self.raw_mfe_classifier is not None:
                    return self.forward_safety_mfe_heads(
                        x, context, relation_history_x=relation_history_x
                    )[1]
                if self.conditional_mfe_classifier is not None:
                    return self.forward_safety_conditional_mfe_heads(x, context)[1]
                primary_input, _shared = self._encoded_for_heads(x, context)
                return self.classifier(primary_input)
            if head in {"conditional_safety", "safety"}:
                return self.forward_conditional_heads(x, context)[1]
            if head in {"conditional_both", "both"}:
                if self.raw_mfe_classifier is not None:
                    safety_logits, raw_mfe_logits = self.forward_safety_mfe_heads(
                        x, context, relation_history_x=relation_history_x
                    )
                    return torch.cat([safety_logits, raw_mfe_logits], dim=1)
                if self.conditional_mfe_classifier is not None:
                    safety_logits, conditional_mfe_logits = self.forward_safety_conditional_mfe_heads(x, context)
                    return torch.cat([safety_logits, conditional_mfe_logits], dim=1)
                primary_logits, conditional_logits = self.forward_conditional_heads(x, context)
                return torch.cat([primary_logits, conditional_logits], dim=1)
            if head in {"tri_head", "safety_raw_mfe_hmhs", "all_three"}:
                safety_logits, raw_mfe_logits, joint_hmhs_logits = self.forward_safety_raw_mfe_hmhs_heads(x, context)
                return torch.cat([safety_logits, raw_mfe_logits, joint_hmhs_logits], dim=1)
            if head in {"raw_safety", "safety_condition"}:
                if self.raw_mfe_classifier is not None:
                    return self.forward_safety_mfe_heads(
                        x, context, relation_history_x=relation_history_x
                    )[0]
                return self.forward_safety_conditional_mfe_heads(x, context)[0]
            if head in {"conditional_mfe", "raw_mfe", "final"}:
                if self.raw_mfe_classifier is not None:
                    return self.forward_safety_mfe_heads(
                        x, context, relation_history_x=relation_history_x
                    )[1]
                return self.forward_safety_conditional_mfe_heads(x, context)[1]
            if head in {"joint_hmhs", "hmhs"}:
                return self.forward_safety_raw_mfe_hmhs_heads(x, context)[2]
            raise ValueError(f"未知InceptionTime output head: {output_head!r}")

        def forward(self, x, context, *, relation_history_x=None):
            if self.conditional_safety_classifier is not None:
                return self.forward_conditional_heads(x, context)[0]
            if self.raw_mfe_classifier is not None:
                return self.forward_safety_mfe_heads(
                    x, context, relation_history_x=relation_history_x
                )[1]
            if self.conditional_mfe_classifier is not None:
                return self.forward_safety_conditional_mfe_heads(x, context)[1]
            primary_input, _shared = self._encoded_for_heads(x, context)
            return self.classifier(primary_input)

    return InceptionTimeClassifier()


__all__ = ["build_inception_time"]
