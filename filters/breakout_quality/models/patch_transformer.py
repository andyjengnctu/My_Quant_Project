"""Small supervised Patch Transformer for breakout-quality sequences."""

from __future__ import annotations

import math


def build_patch_transformer(nn, torch, *, feature_count: int, context_count: int, spec):
    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("Patch Transformer 9F 必須停用 Dataset event context")

    patch_size = int(spec.patch_transformer_patch_size)
    patch_stride = int(spec.patch_transformer_patch_stride)
    embedding_dim = int(spec.patch_transformer_embedding_dim)
    depth = int(spec.patch_transformer_depth)
    heads = int(spec.patch_transformer_heads)
    mlp_dim = int(spec.patch_transformer_mlp_dim)
    pooling = str(spec.patch_transformer_pooling or "mean").strip().lower()
    positional_encoding = str(
        spec.patch_transformer_positional_encoding or "sinusoidal"
    ).strip().lower()

    if int(feature_count) < 1:
        raise ValueError("Patch Transformer feature_count 必須 >= 1")
    if patch_size < 1 or patch_stride != patch_size:
        raise ValueError("Patch Transformer 9F 必須使用正整數非重疊 patch／stride")
    if embedding_dim < 1 or depth < 1 or heads < 1 or mlp_dim < 1:
        raise ValueError("Patch Transformer dimensions／depth／heads 必須是正整數")
    if embedding_dim % heads != 0:
        raise ValueError("Patch Transformer embedding dim 必須可被 attention heads 整除")
    if pooling != "mean":
        raise ValueError("Patch Transformer 9F pooling 固定為 mean")
    if positional_encoding != "sinusoidal":
        raise ValueError("Patch Transformer 9F positional encoding 固定為 sinusoidal")

    def sinusoidal_position_encoding(length: int, *, device, dtype):
        positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dim)
        )
        encoding = torch.zeros(
            (length, embedding_dim), device=device, dtype=torch.float32
        )
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        return encoding.to(dtype=dtype)

    class PatchTransformerClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_count = int(feature_count)
            self.patch_size = patch_size
            self.patch_stride = patch_stride
            self.patch_projection = nn.Linear(
                self.patch_size * self.feature_count,
                embedding_dim,
            )
            self.patch_normalization = nn.LayerNorm(embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=float(spec.dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth,
                enable_nested_tensor=False,
            )
            self.output_normalization = nn.LayerNorm(embedding_dim)
            self.dropout = nn.Dropout(float(spec.dropout))
            self.classifier = nn.Linear(embedding_dim, 2)

        def forward(self, x, context):
            del context
            if x.ndim != 3:
                raise ValueError(f"Patch Transformer input 必須是 [B,T,F]: {tuple(x.shape)}")
            batch_size, sequence_length, input_features = x.shape
            if int(input_features) != self.feature_count:
                raise ValueError(
                    "Patch Transformer feature_count 不一致: "
                    f"expected={self.feature_count}, actual={int(input_features)}"
                )
            if int(sequence_length) < self.patch_size or int(sequence_length) % self.patch_size != 0:
                raise ValueError(
                    "Patch Transformer sequence length 必須可被非重疊 patch size 整除: "
                    f"sequence_length={int(sequence_length)}, patch_size={self.patch_size}"
                )
            patch_count = int(sequence_length) // self.patch_size
            patches = x.contiguous().reshape(
                int(batch_size),
                patch_count,
                self.patch_size * self.feature_count,
            )
            tokens = self.patch_normalization(self.patch_projection(patches))
            tokens = tokens + sinusoidal_position_encoding(
                patch_count,
                device=tokens.device,
                dtype=tokens.dtype,
            ).unsqueeze(0)
            encoded = self.encoder(tokens)
            pooled = self.output_normalization(encoded).mean(dim=1)
            return self.classifier(self.dropout(pooled))

    return PatchTransformerClassifier()


__all__ = ["build_patch_transformer"]
