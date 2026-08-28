"""Active Patch-Transformer tri-head architecture for MR-13Z Joint-Min research."""

from __future__ import annotations

import math


def build_patch_token_joint_min(nn, torch, *, feature_count: int, context_count: int, spec):
    """Build MR-13Z from the frozen historical 9F Patch-Transformer trunk recipe.

    Marginal heads consume the mean of the normalized encoded patch tokens. The
    Joint-Min head consumes a scalar-attention weighted sum over those same patch
    tokens. The current tri-head surface is deliberately separate from the legacy
    9F binary classifier so historical reconstruction remains isolated.
    """

    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("MR-13Z Patch Transformer 必須停用 Dataset event context")

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
    dropout = float(spec.dropout)

    if int(feature_count) < 1:
        raise ValueError("MR-13Z Patch Transformer feature_count 必須 >= 1")
    if patch_size < 1 or patch_stride != patch_size:
        raise ValueError("MR-13Z 必須使用historical 9F正整數非重疊 patch／stride")
    if embedding_dim < 1 or depth < 1 or heads < 1 or mlp_dim < 1:
        raise ValueError("MR-13Z Patch Transformer dimensions／depth／heads 必須是正整數")
    if embedding_dim % heads != 0:
        raise ValueError("MR-13Z Patch Transformer embedding dim 必須可被 attention heads 整除")
    if pooling != "mean":
        raise ValueError("MR-13Z marginal pooling固定沿用historical mean semantics")
    if positional_encoding != "sinusoidal":
        raise ValueError("MR-13Z positional encoding固定沿用historical sinusoidal")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("MR-13Z Patch Transformer dropout 必須介於 [0, 1)")
    if int(spec.head_width or embedding_dim) != embedding_dim:
        raise ValueError("MR-13Z joint MLP hidden width 必須等於Patch Transformer latent width")

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

    class PatchTokenJointMinClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_count = int(feature_count)
            self.patch_size = patch_size
            self.patch_stride = patch_stride
            # Keep historical 9F trunk module creation order/names so Seed42 can
            # reconstruct the legacy trunk tensor-for-tensor.
            self.patch_projection = nn.Linear(
                self.patch_size * self.feature_count,
                embedding_dim,
            )
            self.patch_normalization = nn.LayerNorm(embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
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

            self.raw_safety_classifier = nn.Linear(embedding_dim, 2)
            self.conditional_mfe_classifier = nn.Linear(embedding_dim + 1, 2)
            self.joint_hmhs_classifier = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, 2),
            )
            self.joint_attention_scorer = nn.Conv1d(
                embedding_dim, 1, kernel_size=1, bias=True
            )

        def encode_token_map(self, x):
            if x.ndim != 3:
                raise ValueError(
                    f"MR-13Z Patch Transformer input 必須是 [B,T,F]: {tuple(x.shape)}"
                )
            batch_size, sequence_length, input_features = x.shape
            if int(input_features) != self.feature_count:
                raise ValueError(
                    "MR-13Z Patch Transformer feature_count 不一致: "
                    f"expected={self.feature_count}, actual={int(input_features)}"
                )
            if (
                int(sequence_length) < self.patch_size
                or int(sequence_length) % self.patch_size != 0
            ):
                raise ValueError(
                    "MR-13Z sequence length 必須可被historical非重疊patch size整除: "
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
            return self.output_normalization(self.encoder(tokens))

        def encode(self, x):
            return torch.mean(self.encode_token_map(x), dim=1)

        def joint_attention_weights(self, x):
            token_map = self.encode_token_map(x)
            logits = self.joint_attention_scorer(token_map.transpose(1, 2)).squeeze(1)
            return torch.softmax(logits.float(), dim=1).to(token_map.dtype)

        def _joint_attention_pool(self, token_map):
            logits = self.joint_attention_scorer(token_map.transpose(1, 2)).squeeze(1)
            weights = torch.softmax(logits.float(), dim=1).to(token_map.dtype)
            return torch.sum(token_map * weights.unsqueeze(2), dim=1)

        def forward_safety_conditional_mfe_heads(self, x, context):
            del context
            shared_encoded = torch.mean(self.encode_token_map(x), dim=1)
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
            token_map = self.encode_token_map(x)
            shared_encoded = torch.mean(token_map, dim=1)
            joint_encoded = self._joint_attention_pool(token_map)
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
            raise ValueError(f"未知Patch Transformer Joint-Min output head: {output_head!r}")

        def forward(self, x, context):
            return self.forward_safety_conditional_mfe_heads(x, context)[1]

    return PatchTokenJointMinClassifier()


__all__ = ["build_patch_token_joint_min"]
