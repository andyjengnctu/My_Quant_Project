"""TS2Vec-style temporal encoder and frozen linear classifier."""

from __future__ import annotations

from typing import Mapping


def _same_padding(kernel_size: int, dilation: int) -> int:
    effective = (int(kernel_size) - 1) * int(dilation)
    if effective % 2 != 0:
        raise ValueError("TS2Vec kernel/dilation 必須產生偶數 same padding")
    return effective // 2


def build_ts2vec_encoder(nn, torch, *, feature_count: int, spec):
    hidden_dims = int(spec.ts2vec_hidden_dims or 0)
    output_dims = int(spec.ts2vec_output_dims or 0)
    depth = int(spec.ts2vec_depth or 0)
    kernel_size = int(spec.kernel_size)
    dropout = float(spec.dropout)
    if feature_count < 1 or hidden_dims < 1 or output_dims < 1 or depth < 1:
        raise ValueError("TS2Vec encoder spec 不合法")

    class ResidualDilatedBlock(nn.Module):
        def __init__(self, dilation: int):
            super().__init__()
            padding = _same_padding(kernel_size, dilation)
            self.conv1 = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.conv2 = nn.Conv1d(
                hidden_dims,
                hidden_dims,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
            )
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.GELU()

        def forward(self, x):
            residual = x
            x = self.conv1(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.activation(x)
            x = self.dropout(x)
            return x + residual

    class Ts2VecEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Conv1d(feature_count, hidden_dims, kernel_size=1)
            self.blocks = nn.ModuleList(
                [ResidualDilatedBlock(2**index) for index in range(depth)]
            )
            self.output_projection = nn.Conv1d(hidden_dims, output_dims, kernel_size=1)

        def forward(self, sequence, *, temporal_mask=None):
            if sequence.ndim != 3:
                raise ValueError(f"TS2Vec sequence 必須是 [batch, feature, time]: {sequence.shape}")
            if temporal_mask is not None:
                if temporal_mask.ndim != 2 or temporal_mask.shape != (
                    sequence.shape[0],
                    sequence.shape[2],
                ):
                    raise ValueError("TS2Vec temporal_mask shape 不一致")
                sequence = sequence.masked_fill(~temporal_mask[:, None, :], 0.0)
            x = self.input_projection(sequence)
            for block in self.blocks:
                x = block(x)
            return self.output_projection(x)

    return Ts2VecEncoder()


def build_ts2vec_frozen_linear(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    if bool(spec.use_dataset_context):
        raise ValueError("TS2Vec frozen linear 不得使用 dataset context")
    encoder = build_ts2vec_encoder(nn, torch, feature_count=feature_count, spec=spec)
    output_dims = int(spec.ts2vec_output_dims or 0)

    class Ts2VecFrozenLinearClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(output_dims, 2)
            if pretrained_encoder_state is not None:
                self.encoder.load_state_dict(dict(pretrained_encoder_state), strict=True)
            self.freeze_encoder()

        def freeze_encoder(self) -> None:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.encoder.eval()
            return self

        def forward(self, sequence, context):
            del context
            if sequence.ndim != 3:
                raise ValueError(
                    f"TS2Vec classifier sequence 必須是 [batch,time,feature]: {sequence.shape}"
                )
            if int(sequence.shape[2]) != int(feature_count):
                raise ValueError(
                    "TS2Vec classifier feature_count 不一致: "
                    f"expected={feature_count}, actual={sequence.shape[2]}"
                )
            encoder_input = sequence.transpose(1, 2)
            with torch.no_grad():
                temporal = self.encoder(encoder_input)
                representation = torch.amax(temporal, dim=2)
            return self.classifier(representation)

    return Ts2VecFrozenLinearClassifier()


def instance_contrastive_loss(torch, z1, z2):
    batch_size, time_steps, _channels = z1.shape
    if batch_size <= 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)
    similarity = torch.matmul(z, z.transpose(1, 2))
    logits = similarity.masked_fill(
        torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)[None, :, :],
        float("-inf"),
    )
    log_prob = torch.log_softmax(logits, dim=-1)
    anchors = torch.arange(batch_size, device=z.device)
    positive_a = -log_prob[:, anchors, anchors + batch_size]
    positive_b = -log_prob[:, anchors + batch_size, anchors]
    return 0.5 * (positive_a.mean() + positive_b.mean())


def temporal_contrastive_loss(torch, z1, z2):
    batch_size, time_steps, _channels = z1.shape
    if time_steps <= 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)
    similarity = torch.matmul(z, z.transpose(1, 2))
    logits = similarity.masked_fill(
        torch.eye(2 * time_steps, device=z.device, dtype=torch.bool)[None, :, :],
        float("-inf"),
    )
    log_prob = torch.log_softmax(logits, dim=-1)
    anchors = torch.arange(time_steps, device=z.device)
    positive_a = -log_prob[:, anchors, anchors + time_steps]
    positive_b = -log_prob[:, anchors + time_steps, anchors]
    return 0.5 * (positive_a.mean() + positive_b.mean())


def hierarchical_contrastive_loss(
    torch,
    z1,
    z2,
    *,
    alpha: float,
    temporal_unit: int,
):
    import torch.nn.functional as F

    if z1.shape != z2.shape or z1.ndim != 3:
        raise ValueError("TS2Vec contrastive representations shape 必須一致且為 [batch,time,channel]")
    resolved_alpha = float(alpha)
    if not 0.0 <= resolved_alpha <= 1.0:
        raise ValueError("TS2Vec alpha 必須介於 0 與 1")
    resolved_temporal_unit = int(temporal_unit)
    if resolved_temporal_unit < 0:
        raise ValueError("TS2Vec temporal_unit 必須 >= 0")

    loss = z1.new_tensor(0.0)
    depth = 0
    while z1.shape[1] > 1:
        loss = loss + resolved_alpha * instance_contrastive_loss(torch, z1, z2)
        if depth >= resolved_temporal_unit:
            loss = loss + (1.0 - resolved_alpha) * temporal_contrastive_loss(torch, z1, z2)
        depth += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.shape[1] == 1:
        loss = loss + resolved_alpha * instance_contrastive_loss(torch, z1, z2)
        depth += 1
    return loss / max(depth, 1)


__all__ = [
    "build_ts2vec_encoder",
    "build_ts2vec_frozen_linear",
    "hierarchical_contrastive_loss",
    "instance_contrastive_loss",
    "temporal_contrastive_loss",
]
