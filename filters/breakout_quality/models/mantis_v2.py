"""Frozen MantisV2 encoder with a single supervised linear probe."""

from __future__ import annotations

from typing import Mapping

from filters.breakout_quality.mantis_contract import (
    MANTIS_INSTALL_HINT,
    MANTIS_PACKAGE_NAME,
    MANTIS_PACKAGE_VERSION,
    require_mantis_v2_class,
)


MANTIS_ENCODER_FORWARD_CHUNK_SIZE = 1024


def _build_encoder(spec):
    MantisV2 = require_mantis_v2_class()
    encoder = MantisV2(
        hidden_dim=int(spec.mantis_hidden_dim or 0),
        num_patches=int(spec.mantis_num_patches or 0),
        kernel_size=int(spec.kernel_size),
        scalar_scales=None,
        hidden_dim_scalar_enc=int(spec.mantis_scalar_hidden_dim or 0),
        epsilon_scalar_enc=float(spec.mantis_scalar_epsilon or 0.0),
        transf_depth=int(spec.mantis_transformer_depth or 0),
        transf_num_heads=int(spec.mantis_transformer_heads or 0),
        transf_mlp_dim=int(spec.mantis_transformer_mlp_dim or 0),
        transf_dim_head=int(spec.mantis_transformer_dim_head or 0),
        transf_dropout=float(spec.dropout),
        return_transf_layer=int(spec.mantis_return_transformer_layer or 0),
        output_token=str(spec.mantis_output_token or ""),
        device="cpu",
        pre_training=False,
    )
    encoder.remove_transf_layers()
    return encoder


def build_mantis_v2_frozen_linear(
    nn,
    torch,
    *,
    feature_count: int,
    context_count: int,
    spec,
    pretrained_encoder_state: Mapping[str, object] | None = None,
):
    del context_count
    if bool(spec.use_dataset_context):
        raise ValueError("MantisV2 frozen linear 不得使用 dataset context")
    if int(feature_count) < 1:
        raise ValueError("MantisV2 feature_count 必須 >= 1")
    input_length = int(spec.mantis_input_length or 0)
    num_patches = int(spec.mantis_num_patches or 0)
    embedding_dim = int(spec.mantis_embedding_dim or 0)
    if input_length < 1 or num_patches < 1 or input_length % num_patches != 0:
        raise ValueError("MantisV2 input_length 必須是 num_patches 的正整數倍")
    if embedding_dim < 1:
        raise ValueError("MantisV2 embedding_dim 必須 >= 1")

    encoder = _build_encoder(spec)

    class MantisV2FrozenLinearClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(int(feature_count) * embedding_dim, 2)
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
                    f"MantisV2 sequence 必須是 [batch,time,feature]: {sequence.shape}"
                )
            if int(sequence.shape[2]) != int(feature_count):
                raise ValueError(
                    "MantisV2 feature_count 不一致: "
                    f"expected={feature_count}, actual={sequence.shape[2]}"
                )
            batch_size, _time_steps, channel_count = sequence.shape
            encoder_input = sequence.transpose(1, 2).reshape(
                int(batch_size) * int(channel_count), 1, int(sequence.shape[1])
            )
            encoded_chunks = []
            with torch.no_grad():
                for start in range(0, int(encoder_input.shape[0]), MANTIS_ENCODER_FORWARD_CHUNK_SIZE):
                    encoder_chunk = encoder_input[
                        start : start + MANTIS_ENCODER_FORWARD_CHUNK_SIZE
                    ]
                    if int(encoder_chunk.shape[2]) != input_length:
                        encoder_chunk = torch.nn.functional.interpolate(
                            encoder_chunk,
                            size=input_length,
                            mode="linear",
                            align_corners=False,
                        )
                    encoded_chunk = self.encoder(encoder_chunk)
                    if encoded_chunk.ndim != 2 or int(encoded_chunk.shape[1]) != embedding_dim:
                        raise ValueError(
                            "MantisV2 encoder output shape 不一致: "
                            f"expected=(*,{embedding_dim}), actual={tuple(encoded_chunk.shape)}"
                        )
                    encoded_chunks.append(encoded_chunk)
            if not encoded_chunks:
                raise ValueError("MantisV2 encoder input 不得為空")
            encoded = torch.cat(encoded_chunks, dim=0)
            representation = encoded.reshape(int(batch_size), int(channel_count) * embedding_dim)
            return self.classifier(representation)

    return MantisV2FrozenLinearClassifier()


__all__ = [
    "MANTIS_ENCODER_FORWARD_CHUNK_SIZE",
    "MANTIS_INSTALL_HINT",
    "MANTIS_PACKAGE_NAME",
    "MANTIS_PACKAGE_VERSION",
    "build_mantis_v2_frozen_linear",
    "require_mantis_v2_class",
]
