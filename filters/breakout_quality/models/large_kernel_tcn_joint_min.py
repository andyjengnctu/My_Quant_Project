"""Active large-kernel TCN tri-head architecture for MR-13Y Joint-Min research."""

from __future__ import annotations


def build_modern_tcn_joint_min(nn, torch, *, feature_count: int, context_count: int, spec):
    """Build MR-13Y using the frozen historical 9B ModernTCN trunk recipe.

    The trunk is a large-kernel depthwise residual TCN. Marginal heads use global
    average pooling; the Joint-Min head uses scalar temporal attention pooling.
    Head widths follow the trunk latent width so no independent capacity knob or
    projection adapter is introduced.
    """

    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("MR-13Y ModernTCN 必須停用 Dataset event context")

    depth = int(spec.modern_tcn_depth)
    channels = int(spec.modern_tcn_channels)
    kernel_size = int(spec.modern_tcn_kernel_size)
    expansion_ratio = int(spec.modern_tcn_expansion_ratio)
    dropout = float(spec.dropout)
    if depth < 1 or channels < 1 or expansion_ratio < 1:
        raise ValueError("MR-13Y ModernTCN spec 必須使用正整數 depth／channels／expansion ratio")
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("MR-13Y ModernTCN large kernel 必須是 >=3 的奇數")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("MR-13Y ModernTCN dropout 必須介於 [0, 1)")
    if int(spec.head_width or channels) != channels:
        raise ValueError("MR-13Y joint MLP hidden width 必須等於ModernTCN latent width")

    expanded_channels = channels * expansion_ratio

    class LargeKernelTCNBlock(nn.Module):
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

    class ModernTCNJointMinClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv1d(int(feature_count), channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            self.blocks = nn.ModuleList([LargeKernelTCNBlock() for _ in range(depth)])
            self.raw_safety_classifier = nn.Linear(channels, 2)
            self.conditional_mfe_classifier = nn.Linear(channels + 1, 2)
            self.joint_hmhs_classifier = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, 2),
            )
            self.joint_attention_scorer = nn.Conv1d(
                channels, 1, kernel_size=1, bias=True
            )

        def encode_feature_map(self, x):
            z = self.stem(x.transpose(1, 2))
            for block in self.blocks:
                z = block(z)
            return z

        def encode(self, x):
            return torch.mean(self.encode_feature_map(x), dim=2)

        def joint_attention_weights(self, x):
            feature_map = self.encode_feature_map(x)
            logits = self.joint_attention_scorer(feature_map).squeeze(1)
            return torch.softmax(logits.float(), dim=1).to(feature_map.dtype)

        def _joint_attention_pool(self, feature_map):
            logits = self.joint_attention_scorer(feature_map).squeeze(1)
            weights = torch.softmax(logits.float(), dim=1).to(feature_map.dtype)
            return torch.sum(feature_map * weights.unsqueeze(1), dim=2)

        def forward_safety_conditional_mfe_heads(self, x, context):
            del context
            shared_encoded = torch.mean(self.encode_feature_map(x), dim=2)
            safety_logits = self.raw_safety_classifier(shared_encoded)
            safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
            safety_context = (
                safety_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
            )
            raw_mfe_logits = self.conditional_mfe_classifier(
                torch.cat([shared_encoded, safety_context], dim=1)
            )
            return safety_logits, raw_mfe_logits

        def forward_safety_raw_mfe_hmhs_heads(self, x, context):
            del context
            feature_map = self.encode_feature_map(x)
            shared_encoded = torch.mean(feature_map, dim=2)
            joint_encoded = self._joint_attention_pool(feature_map)
            safety_logits = self.raw_safety_classifier(shared_encoded)
            safety_probability = torch.softmax(safety_logits.float(), dim=1)[:, 1]
            safety_context = (
                safety_probability.detach().to(shared_encoded.dtype).unsqueeze(1)
            )
            raw_mfe_logits = self.conditional_mfe_classifier(
                torch.cat([shared_encoded, safety_context], dim=1)
            )
            joint_logits = self.joint_hmhs_classifier(joint_encoded)
            return safety_logits, raw_mfe_logits, joint_logits

        def forward_output_head(self, x, context, output_head: str):
            head = str(output_head).strip().lower()
            if head in {"primary", "mfe", "primary_mfe", "conditional_mfe", "final"}:
                return self.forward_safety_conditional_mfe_heads(x, context)[1]
            if head in {"conditional_both", "both"}:
                safety_logits, raw_mfe_logits = self.forward_safety_conditional_mfe_heads(
                    x, context
                )
                return torch.cat([safety_logits, raw_mfe_logits], dim=1)
            if head in {"tri_head", "safety_raw_mfe_hmhs", "all_three"}:
                safety_logits, raw_mfe_logits, joint_logits = (
                    self.forward_safety_raw_mfe_hmhs_heads(x, context)
                )
                return torch.cat([safety_logits, raw_mfe_logits, joint_logits], dim=1)
            if head in {"raw_safety", "safety_condition", "safety"}:
                return self.forward_safety_conditional_mfe_heads(x, context)[0]
            if head in {"joint_hmhs", "hmhs"}:
                return self.forward_safety_raw_mfe_hmhs_heads(x, context)[2]
            raise ValueError(f"未知ModernTCN Joint-Min output head: {output_head!r}")

        def forward(self, x, context):
            return self.forward_safety_conditional_mfe_heads(x, context)[1]

    return ModernTCNJointMinClassifier()


__all__ = ["build_modern_tcn_joint_min"]
